# -*- coding: utf-8 -*-
"""GPU 后端 —— 两路由中"结构路由 H"的底座，也是全部可训练基线的公共训练器。

## 为什么必须联合训练，不能事后 Procrustes

本项目 2026-09-27 实测（`.workbuddy/diag_struct.py`）：

    TransE 单独训练后在自身三元组上的 filtered link-prediction MRR = 0.15–0.29
    （随机水平约 1e-4）→ **嵌入本身是学好的**；
    但两侧**各自独立**训练后，用 4,050 条种子做正交 Procrustes，Hits@1 只有 0.005
    （候选集 105,889）→ **对齐失败**。

原因是 KG 嵌入空间**不满足近似等距假设**：TransE 的目标
‖h+r−t‖→0 在任意正交变换下不保持不变（关系向量 r 会被转走），因此两次独立训练
收敛到的两个空间之间**不存在**一个能把语义对应点对齐的正交映射。词向量上的
"跨语言同构"经验对 KG 嵌入不成立 —— 这正是 MTransE（Chen et al., IJCAI 2017）
必须"两个 KG 一起训练、用对齐损失把两个空间绑在一起"的原因。

本模块据此实现三件东西：

  1. `train_joint`  —— MTransE 式联合训练：TransE 损失（两侧）＋ 对齐损失
     ‖E_L[s_l]W − E_R[s_r]‖² ＋ 正交正则 ‖WᵀW − I‖²，实体向量每步投影到单位球；
  2. `JointEmb`     —— 训练产物（两侧实体/关系矩阵 + 对齐映射 W + 设备与耗时）；
  3. `device_report` —— 训练设备的自述，供第 5 章"运行环境"一节引用。

## 设备

`torch.device("cuda")` 存在即用 GPU，否则回落到 CPU。**同一份代码**在两种设备上
跑，只是 `dim/epochs` 与批量大小不同 —— 因此本机跑小规模、腾讯云 GPU 跑完整规模，
两条结果来自同一条代码路径，不存在"两套实现"的隐患。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:                                    # pragma: no cover
    _HAS_TORCH = False


def has_torch() -> bool:
    return _HAS_TORCH


def get_device(prefer: str = "auto"):
    """返回 torch.device。`prefer` ∈ {"auto","cuda","cpu"}。"""
    if not _HAS_TORCH:
        return None
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "cuda":
        raise RuntimeError("请求了 cuda 但 torch.cuda.is_available() 为 False；"
                           "请在腾讯云 GPU 实例上装 CUDA 版 torch，或改用 --device cpu")
    return torch.device("cpu")


def _env_report() -> dict:
    """**环境中立**字段：与 torch 是否可用无关（第 5 章"运行环境"一节的必填项）。

    单独拆出来是因为"跑在什么环境上"与"用不用 GPU"是两件事：前者决定结果文件能否
    被别人复现，后者只决定速度。上游（`run_experiment --device cpu`）与云端
    （`--device cuda`）都必须带上这些字段，否则第 5 章的环境行写不全。
    """
    import os
    import platform
    import sys
    info = {"python": sys.version.split()[0], "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(), "machine": platform.machine(),
            "hostname": platform.node()}
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:                                    # noqa: BLE001
        info["scipy"] = None
    return info


def device_report(prefer: str = "auto") -> dict:
    """设备自述：写入结果文件，供第 5 章复现实验环境。

    返回两部分拼接：
      · **环境中立字段**（python / numpy / scipy / vCPU / 平台 / 主机名）—— 见
        `_env_report`，无 torch 时同样存在；
      · **训练设备字段**（backend / torch / device / device_name / cuda …）。

    本机与云端两份结果文件的差异**应只出现在第二类字段上**。这不会污染结论：
    云端与本地产物的逐格核对脚本 把 `env.*` 整体归类为"运行环境（预期不同）"，
    不计入"待人工复核"格数。
    """
    info = _env_report()
    if not _HAS_TORCH:
        info.update({"backend": "numpy", "torch": None, "device": "cpu",
                     "device_name": "CPU（无 torch，numpy 路径）", "cuda": None})
        return info
    dev = get_device(prefer)
    info.update({"backend": "torch", "torch": torch.__version__,
                 "device": str(dev), "device_name": "CPU", "cuda": None})
    if dev.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(dev)
        info["cuda"] = torch.version.cuda
        info["cuda_capability"] = ".".join(map(str, torch.cuda.get_device_capability(dev)))
        info["cuda_mem_GB"] = round(
            torch.cuda.get_device_properties(dev).total_memory / 1024 ** 3, 1)
    return info


# ================================================================ 训练产物
@dataclass
class JointEmb:
    """联合训练的产物。所有矩阵都是 `numpy`，便于下游用 numpy 打分。"""
    EL: np.ndarray                     # (n_L, d) 单位球上
    ER: np.ndarray                     # (n_R, d) 单位球上
    RL: np.ndarray
    RR: np.ndarray
    W: np.ndarray                      # (d, d) 正交对齐映射：E_L W ≈ E_R
    align: str
    dim: int
    train_time: float
    loss_hist: list = field(default_factory=list)
    align_hist: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---- 供"路由 H"使用的双侧可比表征 ----
    def views(self):
        ZL = self.EL @ self.W
        return _l2(ZL), _l2(self.ER)


def _l2(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (M / n).astype(np.float32)


def _procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """正交 Procrustes 的 SVD 闭式解（用于训练前给 W 一个好初值）。"""
    U, _, Vt = np.linalg.svd(X.T.astype(np.float64) @ Y.astype(np.float64))
    return (U @ Vt).astype(np.float32)


# ================================================================ 联合训练
def train_joint(kl, kr, seeds: np.ndarray,
                dim: int = 128, epochs: int = 60, lr: float = 1e-3,
                n_neg: int = 4, margin: float = 1.0, batch: int = 4096,
                align: str = "ortho", lam_align: float = 1.0,
                align_margin: float = 0.4, n_align_neg: int = 4,
                lam_reg: float = 0.05, seed: int = 2026,
                device=None, warm_start: bool = True,
                monitor=None, monitor_every: int = 1,
                max_minutes: float | None = None, verbose: bool = False) -> JointEmb:
    """MTransE 式联合训练。

    参数
    ----
    kl, kr     : 两侧的 `KG`（关系词表各自独立）
    seeds      : (n_s, 2) 种子对齐下标对 —— **训练期唯一用到的监督**
    align      : "ortho"（W 正交，每步重投影）或 "linear"（自由线性映射）
    lam_align  : 对齐损失权重
    align_margin, n_align_neg
               : 对齐用**边际排序损失**（而非平方误差）—— 见下面"为什么不用 MSE"
    lam_reg    : 关系向量的 L2 正则（TransE 原文的 ‖r‖ 约束）
    monitor    : 可选回调 `f(epoch, emb_snapshot) -> float`，用于逐轮观察验证指标
    max_minutes: 训练墙上时间上限，超时提前收尾（云端调参时防止失控）

    **为什么对齐损失必须带负采样（2026-09-27 实测教训）**
    最初写成 ‖E_L[s_l]W − E_R[s_r]‖²：8 轮后**种子对齐余弦 = 0.9971，但测试
    Hits@1 = 0.0000**。原因是该目标存在平凡极小 —— 把两侧的种子实体都收缩到各自
    的同一个点即可让损失归零（单位球约束只挡范数、挡不住角度塌缩），而 W 退化成
    任意的正交矩阵，非种子实体因此全无定位。改为边际排序式对齐
    `max(0, γ + cos(d_l, d_r⁻) − cos(d_l, d_r))`，并同时用**批内打乱**与**全库随机**
    两类负例，塌缩即被阻断。
    """
    if not _HAS_TORCH:
        raise RuntimeError("torch 不可用；请 pip install torch，或改用 numpy 路径")
    dev = device or get_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    tl = torch.as_tensor(np.asarray(kl.rel_triples, dtype=np.int64), device=dev)
    tr = torch.as_tensor(np.asarray(kr.rel_triples, dtype=np.int64), device=dev)
    sl_np = seeds[:, 0].astype(np.int64)
    sl = torch.as_tensor(sl_np, device=dev)
    sr = torch.as_tensor(seeds[:, 1].astype(np.int64), device=dev)

    nL, nR = kl.n, kr.n
    eL = (torch.randn(nL, dim, device=dev) * 0.1).requires_grad_(True)
    eR = (torch.randn(nR, dim, device=dev) * 0.1).requires_grad_(True)
    rL = (torch.randn(max(kl.n_rel, 1), dim, device=dev) * 0.1).requires_grad_(True)
    rR = (torch.randn(max(kr.n_rel, 1), dim, device=dev) * 0.1).requires_grad_(True)

    with torch.no_grad():
        zl = torch.nn.functional.normalize(eL[sl], dim=1)
        zr = torch.nn.functional.normalize(eR[sr], dim=1)
        if warm_start and len(sl) >= dim:
            W0 = _procrustes(zl.cpu().numpy(), zr.cpu().numpy())
        else:
            W0 = np.eye(dim, dtype=np.float32)
    W = torch.as_tensor(W0, device=dev).requires_grad_(True)

    params = [eL, eR, rL, rR, W]
    opt = torch.optim.Adam(params, lr=lr)
    nLt, nRt = len(tl), len(tr)
    if nLt == 0 and nRt == 0:
        raise ValueError("两侧都无关系三元组，无法做结构联合训练")
    loss_hist, align_hist, t0 = [], [], time.time()

    n_it_l = max((nLt + batch - 1) // batch, 1) if nLt else 0
    n_it_r = max((nRt + batch - 1) // batch, 1) if nRt else 0
    n_iter = max(n_it_l, n_it_r)
    for ep in range(epochs):
        pl = rng.permutation(nLt) if nLt else None
        pr = rng.permutation(nRt) if nRt else None
        ep_loss, ep_al, nb = 0.0, 0.0, 0
        for it in range(n_iter):
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=dev)
            if nLt and it < n_it_l:
                b = _take(tl, pl, it * batch, batch)
                loss = loss + _transe_loss(eL, rL, b, nL, n_neg, margin, lam_reg)
            if nRt and it < n_it_r:
                b = _take(tr, pr, it * batch, batch)
                loss = loss + _transe_loss(eR, rR, b, nR, n_neg, margin, lam_reg)
            # ---- 对齐：边际排序损失 + 批内与全库两类负例 ----
            if len(sl):
                k = min(len(sl), 1024)
                sel = torch.as_tensor(rng.choice(len(sl), k, replace=False), device=dev)
                dl = torch.nn.functional.normalize(eL[sl[sel]] @ W, dim=1)
                dr = torch.nn.functional.normalize(eR[sr[sel]], dim=1)
                a_loss = torch.zeros((), device=dev)
                for c in range(n_align_neg):
                    if c % 2 == 0:                       # 批内打乱（难负例）
                        nd = sr[sel][torch.randperm(k, device=dev)]
                    else:                                # 全库随机（易负例）
                        nd = torch.randint(0, nR, (k,), device=dev)
                    dn = torch.nn.functional.normalize(eR[nd], dim=1)
                    a_loss = a_loss + torch.clamp(
                        align_margin + (dl * dn).sum(1) - (dl * dr).sum(1),
                        min=0).mean()
                a_loss = a_loss / max(n_align_neg, 1)
                loss = loss + lam_align * a_loss
                ep_al += float(a_loss.detach())
            loss.backward()
            opt.step()
            # ---- 实体投影到单位球（TransE 的标准约束，每步执行）----
            with torch.no_grad():
                eL.copy_(torch.nn.functional.normalize(eL, dim=1))
                eR.copy_(torch.nn.functional.normalize(eR, dim=1))
                if align == "ortho":
                    u, _s, vh = torch.linalg.svd(W, full_matrices=False)
                    W.copy_(u @ vh)
            ep_loss += float(loss.detach())
            nb += 1
            if max_minutes and (time.time() - t0) / 60 > max_minutes:
                break
        loss_hist.append(ep_loss / max(nb, 1))
        align_hist.append(ep_al / max(nb, 1))
        if verbose:
            print(f"    epoch {ep + 1}/{epochs}  loss={loss_hist[-1]:.4f}"
                  f"  align={align_hist[-1]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if monitor is not None and (ep + 1) % monitor_every == 0:
            snap = JointEmb(EL=eL.detach().cpu().numpy().astype(np.float32),
                            ER=eR.detach().cpu().numpy().astype(np.float32),
                            RL=rL.detach().cpu().numpy().astype(np.float32),
                            RR=rR.detach().cpu().numpy().astype(np.float32),
                            W=W.detach().cpu().numpy().astype(np.float32),
                            align=align, dim=dim, train_time=time.time() - t0)
            monitor(ep + 1, snap)
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            if verbose:
                print(f"    [提前收尾] 达到时间上限 {max_minutes} min", flush=True)
            break

    return JointEmb(EL=eL.detach().cpu().numpy().astype(np.float32),
                    ER=eR.detach().cpu().numpy().astype(np.float32),
                    RL=rL.detach().cpu().numpy().astype(np.float32),
                    RR=rR.detach().cpu().numpy().astype(np.float32),
                    W=W.detach().cpu().numpy().astype(np.float32),
                    align=align, dim=dim, train_time=time.time() - t0,
                    loss_hist=loss_hist, align_hist=align_hist,
                    meta={"epochs_done": len(loss_hist), "epochs_planned": epochs,
                          "n_neg": n_neg, "lr": lr, "batch": batch,
                          "lam_align": lam_align, "align_margin": align_margin,
                          "seeds": int(len(sl)), "device": str(dev)})


def _take(t, perm, start, batch):
    n = len(t)
    idx = perm[start % n: (start % n) + batch]
    if len(idx) == 0:
        idx = perm[:batch]
    return t[torch.as_tensor(idx)]


def _transe_loss(E, R, b, n_e, n_neg, margin, lam_reg):
    """TransE 边际损失（交替腐蚀头/尾，等价于简化版 Bernoulli 采样）。"""
    import torch
    h, r, t = b[:, 0], b[:, 1], b[:, 2]
    n = len(b)
    loss = torch.zeros((), device=E.device)
    for c in range(n_neg):
        corrupt_head = bool(c % 2 == 1)
        src = h if corrupt_head else t
        cand = torch.randint(0, n_e, (n,), device=E.device)
        cand = torch.where(cand == src, (cand + 1) % n_e, cand)
        h2 = cand if corrupt_head else h
        t2 = t if corrupt_head else cand
        pos = E[h] + R[r] - E[t]
        neg = E[h2] + R[r] - E[t2]
        dpos = pos.norm(dim=1)
        dneg = neg.norm(dim=1)
        loss = loss + torch.clamp(margin + dpos - dneg, min=0).mean()
    loss = loss / max(n_neg, 1) + lam_reg * R.norm(dim=1).pow(2).mean()
    return loss
