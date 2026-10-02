# -*- coding: utf-8 -*-
"""设备后端：同一份计算在 CPU（numpy）与 GPU（torch）上执行。

## 为什么需要这一层

第 4 章的构件（熵正则最优传输、纤维掩码、两路由融合、SPS 审计）落地后都是**稠密
矩阵运算**：候选池是 10^5 量级，单块 `chunk × N` 的得分矩阵即 10^2 MB 量级，而
"逐元素比较 + 计数 + 归约"要在这张矩阵上反复做。

2026-09-28 在 15 GB 内存的 CPU 实例上实测：全量 yago3_10 的评测阶段把常驻内存顶到
**15.27 GB**，进程被内核 OOM 杀死（`dmesg` 记为 `Killed process 16537`），约 6 h 55 m
的计算因产物未落盘而全部作废。这类运算恰恰是 GPU 的强项——因此本层把热点下沉到
torch，让同一份算法既能在 CPU 上跑通、也能在 GPU 上跑快。

## 设计原则（三条，改动时必须同时满足）

1. **公开接口仍是 numpy。** `score_block` 等对外函数默认返回 numpy 数组，上层
   （`sps` / `sic` / 统计检验 / 交叉数据集度量）零改动。项目纪律要求"主表与消融
   跑**同一份代码路径**"，因此后端切换只允许改变**在哪里算**，不允许改变**算什么**。
2. **只在热点下沉。** 下沉点只有三处：熵正则最优传输的迭代、大矩阵乘、全候选集上的
   比较/计数。每处的计算量都远大于 CPU↔GPU 的拷贝成本；其余一律保持 numpy。
3. **可逐位回退。** 无 torch、或缺 CUDA、或显式 `--device cpu` 时走 numpy 分支，
   结果与改造前**逐位一致**（由 `code/test_device_parity.py` 断言）。

## 设备选择

    CUSSM_DEVICE 环境变量  >  显式传参  >  auto（有 CUDA 用 CUDA，否则 CPU）

`run_experiment.py --device {auto,cuda,cpu}` 会把解析结果写入环境变量，故子模块
无需逐层传参即可拿到同一设备——这避免了"打分在 GPU、度量在 CPU"这类静默混用。

## 数值口径

* 熵正则最优传输在 **float64** 上迭代（与 CPU 版同一精度），出口转 float32；
  日志域实现的稳定性不依赖后端。
* 比较/计数在 **float32** 上做（得分块本来就是 float32），计数结果为 int64。
* GPU 归约顺序与 CPU 不同，末位可能有 1 ulp 级差异；这不影响排名（项目实测：
  同 seed 两遍在 CPU 上逐位相同，跨设备则需按容差比对）。**任何"逐位相同"的断言
  都只在同一设备内成立**——跨设备的验收口径是"排名与指标一致"。
"""
from __future__ import annotations

import os

import numpy as np

try:                                                    # pragma: no cover
    import torch
    _HAS_TORCH = True
except Exception:                                       # noqa: BLE001
    _HAS_TORCH = False


def lock_precision() -> dict:
    """把浮点精度口径钉死，不依赖 torch 的默认值。返回钉死后的实际取值。

    为什么必须显式做（2026-09-28 实测发现）：torch 的 `allow_tf32` 默认值随版本变动
    ——本机 CPU 版与云端 2.14.0+cu126 上实测 `matmul.allow_tf32=False` 而
    `cudnn.allow_tf32=True`，且代码里此前**从未设置过**。危害有二：

    1. **ViT 的 patch embedding 就是一层 `nn.Conv2d`**（`google/vit-base-patch16-224`
       的 patch 16×16）。cudnn 的 TF32 会把这一层压到 **10 位尾数**，而本章其余
       构件都按真 float32 计算 —— 同一份"GPU 数值"里混了两种精度语义。
    2. 默认值随 torch 版本变，等于让**已发表的读数无法复现**。

    口径：全程真 float32（与 `float32_matmul_precision="highest"` 等价），
    `cudnn.benchmark` 也关掉以免算法自动选择造成逐次差异（eager 模式）。
    """
    if not _HAS_TORCH:
        return {"torch": None}
    out = {}
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        out["matmul.allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    except Exception:                                   # noqa: BLE001
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False
        out["cudnn.allow_tf32"] = torch.backends.cudnn.allow_tf32
    except Exception:                                   # noqa: BLE001
        pass
    try:
        torch.set_float32_matmul_precision("highest")
        out["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except Exception:                                   # noqa: BLE001
        pass
    try:
        torch.backends.cudnn.benchmark = False
        out["cudnn.benchmark"] = torch.backends.cudnn.benchmark
    except Exception:                                   # noqa: BLE001
        pass
    return out


# 导入即钉死：任何入口脚本只要 `import cussm`，精度口径就已确定，不存在"忘设"的路径。
_PRECISION = lock_precision()


# 环境变量名：由 run_experiment.py 写入，子模块读取，保证全流程同一设备。
ENV_KEY = "CUSSM_DEVICE"


def torch_available() -> bool:
    return _HAS_TORCH


def cuda_available() -> bool:
    return bool(_HAS_TORCH and torch.cuda.is_available())


def resolve_name(prefer: str | None = None) -> str:
    """把 {"auto","cuda","cpu"} 解析成 {"cpu","cuda"} 之一。

    解析顺序：显式传参 > 环境变量 > auto。`--device cuda` 在无 CUDA 时**报错而
    不是静默回落**——静默回落会让"GPU 结果"里混入 CPU 读数，属于最坏的一类错误。
    """
    if prefer is None:
        prefer = os.environ.get(ENV_KEY) or "auto"
    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda":
        if not cuda_available():
            raise RuntimeError(
                "请求了 --device cuda，但 torch.cuda.is_available() 为 False。"
                "请确认已装 CUDA 版 torch 且实例有可用 GPU；"
                "若确实要在 CPU 上跑，请显式改用 --device cpu。")
        return "cuda"
    return "cuda" if cuda_available() else "cpu"


def export(prefer: str | None = None) -> str:
    """解析设备并写入环境变量，供子模块共享。返回设备名。"""
    name = resolve_name(prefer)
    os.environ[ENV_KEY] = name
    return name


def describe() -> dict:
    """设备自述，供 `device_report` 与结果文件引用。"""
    name = resolve_name()
    d = {"device": name, "torch": None, "cuda_runtime": None,
         "gpu_name": None, "gpu_memory_gb": None, "compute_capability": None,
         "precision": dict(_PRECISION)}
    if _HAS_TORCH:
        d["torch"] = torch.__version__
        if name == "cuda":
            try:
                p = torch.cuda.get_device_properties(0)
                d["cuda_runtime"] = torch.version.cuda
                d["gpu_name"] = p.name
                d["gpu_memory_gb"] = round(p.total_memory / 1024 ** 3, 1)
                d["compute_capability"] = f"sm_{p.major}{p.minor}"
            except Exception:                            # noqa: BLE001
                pass
    return d


# ============================================================================
# 后端
# ============================================================================
class Backend:
    """一个设备上的张量运算后端。

    方法刻意保持**少而贴调用点**：每一个都对应一处真实热点，避免造出一个
    半吊子线性代数库（那种设计会诱使上层代码照抄 API，反而更难回退）。
    """

    __slots__ = ("name", "is_gpu")

    def __init__(self, name: str | None = None):
        self.name = resolve_name(name)
        self.is_gpu = (self.name == "cuda")

    # ------------------------------------------------------------ 基本转换
    def to_dev(self, a, dtype=None):
        """numpy 数组 -> 设备张量（CPU 后端时原样返回 numpy）。"""
        if not self.is_gpu:
            return np.asarray(a, dtype=dtype) if dtype is not None else np.asarray(a)
        t = torch.as_tensor(np.ascontiguousarray(a))
        if dtype is not None:
            t = t.to(dtype)
        return t.to("cuda", non_blocking=False)

    def to_np(self, t):
        """设备张量 -> numpy（CPU 张量直接转，避免多余的 .cpu() 开销）。"""
        if isinstance(t, np.ndarray):
            return t
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def _f64(self, a):
        return self.to_dev(a, torch.float64) if self.is_gpu else np.asarray(a, np.float64)

    def _f32(self, a):
        return self.to_dev(a, torch.float32) if self.is_gpu else np.asarray(a, np.float32)

    # ------------------------------------------------------------ 热点 1：矩阵乘
    def matmul(self, a, b, dtype=np.float32):
        """`a @ b`。用于两路由的得分块（(m,d) @ (d,N)，d=512、N=10^5 量级）。

        CPU 分支保留在 float32 上算再升位——与改造前 `(NL[idx] @ NR.T).astype(np.float32)`
        的语义一致（numpy 会用 float32 累加）。
        """
        A, B = np.asarray(a), np.asarray(b)
        if not self.is_gpu:
            return (A @ B).astype(dtype)
        ta = self.to_dev(A)
        tb = self.to_dev(B)
        # 梯度不需要，且评测期频繁调用；no_grad 避免 autograd 图累积。
        with torch.no_grad():
            return self.to_np((ta @ tb).to(torch.float32))

    def mix_matmul(self, a1, b1, w1, a2, b2, w2, dtype=np.float32):
        """`w1·(a1@b1) + w2·(a2@b2)` —— 语义路由的加权和。

        与"调两次 `matmul` 再相加"的结果**逐位等价**（float32 累加顺序不变），
        区别只在设备往返次数：两次独立调用会把两个 (m,N) 中间结果各搬回 CPU 一次，
        本方法在设备上完成加权后再搬一次。以 m=512、N=105,889 计，每块省下约
        217 MB × 1 次的 PCIe 传输。
        """
        A1, B1 = np.asarray(a1), np.asarray(b1)
        A2, B2 = np.asarray(a2), np.asarray(b2)
        if not self.is_gpu:
            return (np.float32(w1) * (A1 @ B1).astype(dtype)
                    + np.float32(w2) * (A2 @ B2).astype(dtype)).astype(dtype)
        with torch.no_grad():
            t1 = self.to_dev(A1) @ self.to_dev(B1)
            t2 = self.to_dev(A2) @ self.to_dev(B2)
            out = (float(w1) * t1.to(torch.float32)
                   + float(w2) * t2.to(torch.float32))
        return self.to_np(out).astype(dtype)

    # ------------------------------------------------------------ 热点 2：比较与计数

    def rank_and_ties(self, S, gt, need_tie=False, need_add=True, need_argmax=True):
        """全候选集上的排名统计 —— 旧机 OOM 的直接来源。

        返回 `(n_gt, n_tie, add, amax)`：

            n_gt  : 每行得分**严格大于**真值得分 s_gt 的候选数
            n_tie : 每行得分**等于** s_gt 的候选数（含真值自身）
            add   : 每行中「下标 < 真值下标 且 得分 == s_gt」的候选数
            amax  : 每行 argmax（并列时取下标小者，与 np.argmax 逐位一致）

        语义与 `metrics/core.py` 的原始实现严格对应：

            r_opt = n_gt + 1                    （乐观口径，并列记 rank 1）
            r     = n_gt + add + 1              （模型实际输出口径，与 argmax 一致）

        `add` 的两种实现在语义上等价，选择由后端决定：

        * **CPU**：逐行 `count_nonzero(S[k, :g] == s_gt[k])`。原实现如此，且刻意
          不用 `chunk × N` 的 int64 前缀和（512 × 105,889 × 8 B ≈ 434 MB 会爆内存），
          也不用 `(m,N)` 的 bool 掩码（再省 54 MB/块 × 2）。
        * **GPU**：向量化为 `(eq & (cols < g[:,None])).sum(1)`。显存 16 GB，多物化
          两张 bool 矩阵（各 54 MB）无压力，而省下的 m 次内核启动才是主要收益。
        """
        S = np.asarray(S, dtype=np.float32)
        gt = np.asarray(gt).astype(np.int64).ravel()
        m, N = S.shape

        if not self.is_gpu:
            rows = np.arange(m)
            s_gt = S[rows, gt]
            n_gt = (S > s_gt[:, None]).sum(axis=1)
            n_tie = (S == s_gt[:, None]).sum(axis=1) if need_tie else None
            add = None
            if need_add:
                add = np.empty(m, dtype=np.int64)
                for k in range(m):
                    g = int(gt[k])
                    add[k] = int(np.count_nonzero(S[k, :g] == s_gt[k]))
            amax = S.argmax(axis=1) if need_argmax else None
            return n_gt.astype(np.int64), \
                   (n_tie.astype(np.int64) if n_tie is not None else None), add, amax

        tS = self.to_dev(S)
        tgt = self.to_dev(gt)
        rows = torch.arange(m, device="cuda")
        with torch.no_grad():
            s_gt = tS[rows, tgt]
            n_gt = (tS > s_gt[:, None]).sum(dim=1)
            eq = (tS == s_gt[:, None]) if (need_tie or need_add) else None
            n_tie = eq.sum(dim=1) if need_tie else None
            add = None
            if need_add:
                cols = torch.arange(N, device="cuda")[None, :]
                add = (eq & (cols < tgt[:, None])).sum(dim=1)
            amax = tS.argmax(dim=1) if need_argmax else None
        return (self.to_np(n_gt).astype(np.int64),
                (self.to_np(n_tie).astype(np.int64) if n_tie is not None else None),
                (self.to_np(add).astype(np.int64) if add is not None else None),
                (self.to_np(amax).astype(np.int64) if amax is not None else None))

    def ranks_of_truth(self, S, truth, mask=None, chunk: int = 512) -> np.ndarray:
        """真值排名（1 = 榜首），分块进行，不物化完整排序矩阵。"""
        S = np.asarray(S, dtype=np.float32)
        truth = np.asarray(truth).astype(np.int64).ravel()
        n = S.shape[0]
        out = np.empty(n, dtype=np.int64)
        for i in range(0, n, chunk):
            blk = S[i:i + chunk]
            g = truth[i:i + chunk]
            if mask is not None:
                mb = np.asarray(mask[i:i + chunk])
                blk = np.where(mb, np.float32(-1e30), blk)
            n_gt, _, _, _ = self.rank_and_ties(
                blk, g, need_tie=False, need_add=False, need_argmax=False)
            out[i:i + chunk] = n_gt + 1
        return out

    # ------------------------------------------------------------ 热点 3：短名单
    def topk_indices(self, S, k: int) -> np.ndarray:
        """每行取前 k 大的**下标**（不保证有序，与 `np.argpartition` 同语义）。

        用于粘合路由的候选短名单。`torch.topk` 返回的是降序前 k，其顺序信息
        在调用点被逐行去重后丢弃，故两者可互换。
        """
        S = np.asarray(S, dtype=np.float32)
        k = min(k, S.shape[1])
        if not self.is_gpu:
            return np.argpartition(-S, k - 1, axis=1)[:, :k]
        with torch.no_grad():
            tS = self.to_dev(S)
            _, idx = torch.topk(tS, k, dim=1, largest=True, sorted=False)
        return self.to_np(idx).astype(np.int64)

    # ------------------------------------------------------------ 热点 4：最优传输
    def sinkhorn(self, C, eps: float = 0.05, mask=None,
                 n_iter: int = 80, tol: float = 1e-7):
        """熵正则最优传输（log 域稳定实现），返回软指派 float32。

        `mask` 为 True 表示**禁止**该配对——纤维硬约束在传输**之前**施加，
        而非解完再过滤。这正是 4.2 的"约束在最优点内"在算法上的落点。

        CPU 与 GPU 走**同一条迭代式**（log 域、u/v 交替、按 tol 提前停），
        故收敛路径一致；差异仅来自 float64 归约顺序的末位。

        ## 归一化的真实性质（2026-09-28 实测，勿再按旧注释理解）

        本实现是「行列交替归一化」且**不含边缘约束**：`u` 更新使**行和为 1**
        （在给定 `v` 下），`v` 更新使**列和为 1**（在给定 `u` 下），而循环是以
        `v` 收尾的。于是：

        * **方阵（n1 == n2）**：交替投影存在不动点（双随机矩阵），行和与列和
          都收敛到 1。实测 40×40 与 50×50 在 eps ∈ {0.05, 1, 5}、n_iter=4000 下
          两侧偏差均 ≤ 3.6e-07。
        * **长方形（n1 ≠ n2）**：**不动点不存在**——"每行和为 1"与"每列和为 1"
          同时成立蕴含 `n1 == n2`。实测 60×80 时列和为 1（2.4e-07）而**行和偏差
          恒为 1/3**，且**与迭代次数无关**（80 / 500 / 5000 轮结果逐位相同，
          `max|Δu|` 停滞在 0.28768）。这不是"收敛不足"，是几何上的不可能。

        **对结果无害的理由**：调用方 `CUSSM._route_C` 传的正是长方形代价矩阵
        （`chunk × 短名单`），但它只取 `argsort(log P)` 的**行内相对序**——行缩放
        因子是同一常数，不影响排序。故本性质不改变任何指标。

        若将来需要"真正的最优传输计划"（例如报告传输代价或做锚点约束），
        **必须显式给定边缘约束**（如 a = 1/n1、b = 1/n2），或把 C 补成方阵；
        直接套用当前实现会得到"最后一侧归一化"的结果。
        """
        C = np.asarray(C, dtype=np.float64)
        if not self.is_gpu:
            logK = -C / max(eps, 1e-9)
            if mask is not None:
                logK = np.where(np.asarray(mask), -1e12, logK)
            n1, n2 = logK.shape
            u = np.zeros(n1)
            v = np.zeros(n2)
            for _ in range(n_iter):
                u_new = -_lse_row_np(logK, v)
                v_new = -_lse_col_np(logK, u_new)
                conv = (np.max(np.abs(u_new - u)) < tol
                        and np.max(np.abs(v_new - v)) < tol)
                u, v = u_new, v_new
                if conv:
                    break
            P = np.exp(logK + u[:, None] + v[None, :])
            return P.astype(np.float32)

        with torch.no_grad():
            tC = self.to_dev(C)
            logK = -tC / max(eps, 1e-9)
            if mask is not None:
                logK = torch.where(self.to_dev(np.asarray(mask)), 
                                   torch.tensor(-1e12, dtype=torch.float64,
                                                device="cuda"), logK)
            n1, n2 = logK.shape
            u = torch.zeros(n1, dtype=torch.float64, device="cuda")
            v = torch.zeros(n2, dtype=torch.float64, device="cuda")
            for _ in range(n_iter):
                u_new = -_lse_row_t(logK, v)
                v_new = -_lse_col_t(logK, u_new)
                conv = bool((u_new - u).abs().max() < tol
                            and (v_new - v).abs().max() < tol)
                u, v = u_new, v_new
                if conv:
                    break
            P = torch.exp(logK + u[:, None] + v[None, :])
        return self.to_np(P).astype(np.float32)


# ---------------------------------------------------------------- numpy 内核
def _lse_row_np(logK, v):
    M = logK + v[None, :]
    m = M.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(M - m).sum(axis=1, keepdims=True))).ravel()


def _lse_col_np(logK, u):
    M = logK + u[:, None]
    m = M.max(axis=0, keepdims=True)
    return (m + np.log(np.exp(M - m).sum(axis=0, keepdims=True))).ravel()


# ---------------------------------------------------------------- torch 内核
def _lse_row_t(logK, v):
    M = logK + v[None, :]
    m = M.max(dim=1, keepdim=True).values
    return (m + torch.log(torch.exp(M - m).sum(dim=1, keepdim=True))).ravel()


def _lse_col_t(logK, u):
    M = logK + u[:, None]
    m = M.max(dim=0, keepdim=True).values
    return (m + torch.log(torch.exp(M - m).sum(dim=0, keepdim=True))).ravel()


# ---------------------------------------------------------------- 单例
_CACHE: dict[str, Backend] = {}


def get(prefer: str | None = None) -> Backend:
    """取后端（按名字缓存，避免在循环里反复解析设备）。"""
    name = resolve_name(prefer)
    b = _CACHE.get(name)
    if b is None:
        b = Backend(name)
        _CACHE[name] = b
    return b


def reset_cache() -> None:
    """测试用：切换设备后清缓存。"""
    _CACHE.clear()
