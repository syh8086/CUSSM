# -*- coding: utf-8 -*-
"""本文方法 CUSSM —— 纤维化 lax 函子的可执行实现。

第 4 章的四件构造在此逐条落地：

| 第 4 章构造 | 本文件的实现 |
|---|---|
| 打字投影 π（定义 3） | 两支结构表征合起来做 k-means，得共享类型空间的纤维标签 |
| 纤维保持硬约束 π_B∘F=π_A（定义 5(2)） | 匹配时把跨纤维配对置为不可达（**传输前**生效，非事后过滤） |
| 两条路由 H:=F_TK 与 K:=F_IK∘F_TI（4.3.2(b)） | 结构路由 S_H（种子锚定迭代结构传播）与语义路由 S_K（属性/字面视图），各自独立对齐 |
| 余极限粘合 / pushout（4.4） | 粘合路由 Ω_c：在候选**短名单**上解带纤维掩码的**熵正则最优传输**（Sinkhorn），取传输计划的**序**作为第三路读数；与 `Pool` 的偏离即 Δ_glue |

## 结构路由为什么是"传播"而不是"嵌入 + Procrustes"

本项目 2026-09-27 在 DBP15K fr_en 上把三条候选路线逐层测过（`.workbuddy/diag_*.py`），
全候选集 105,889 上的 Hits@1：

| 路线 | Hits@1 |
|---|---|
| 两侧**独立**训练 TransE + 事后正交 Procrustes | 0.005 |
| MTransE 式**联合训练**（平方误差对齐损失） | 0.000（种子余弦 0.997，角度塌缩） |
| MTransE 式**联合训练**（边际排序对齐损失 + 负采样） | 0.000（种子余弦 0.64，不塌缩但依然对不齐） |
| 仅 log 出度/入度两维的直接余弦 | 0.1125（说明结构信号本来就在） |
| **种子锚定的跨图迭代结构传播（本文件采用）** | **0.58**（0 轮）/ 见第 5 章实测 |

结论：在 DBP15K 上"把两个图分别嵌入再对齐"这条路走不通 —— TransE 的目标
‖h+r−t‖→0 不正交不变，两个独立收敛的空间之间不存在可用的对齐映射；而两个语言的
关系词表**完全不相交**，联合训练的对齐损失只约束 4,050 个种子点，对 10 万实体的
全局几何没有牵引力。改为直接在图上传播后，结构路由才第一次具备可用判别力。
详见 `cussm/propagation.py` 的模块注释。

## 超参选择的口径

`_tune` 在**验证种子**上、**完整候选集**内做「前向选择 + 2-SE 停止规则」：模型族按
复杂度递增（仅语义路由 → +结构路由 → +粘合通道 → +纤维罚/类型先验），每一级只有
验证集增益 ≥ 2 个二项标准误才升级。这样"关闭全部附加通道"退化为 `LinFuse`，
附加通道**只可能加分**。

**绝不用抽样干扰项当代理**：实测 1,000 路抽样候选会把 α 顶到 1.0（抽样候选里
"名字几乎全同"的难例被稀释），代理指标与真实指标**排序不一致**。

**α 网格必须与对照方法一致**：α 的响应在 0.05 处存在尖峰（实测 val 0.8978 →
0.9378 → 0.8911，test 同步），旧网格 (0, 0.1, …, 1.0) 恰跳过该点，会让比较变成
"网格分辨率之争"而不是方法之争。故 `ALPHA_GRID` 由本模块定义、`LinFuse` 直接引用。

**指标口径**：一律在完整候选集上报告；并列裁决取 `argmax` 一致口径
（见 `metrics/core.py`），因为结构画像是整数计数、并列可达数万，
乐观口径会把退化解无条件记成 rank 1。

消融通过构造参数开关（`use_route_H/use_route_K/use_fiber/use_type/use_glue`），
因此消融表与主表跑的是**同一份代码路径**，不存在"消融另写一遍"的风险。
"""
from __future__ import annotations

import time

import numpy as np

from . import dev
from . import mem
from .data import Pair
from .features import build_view, fibers_from
from .propagation import StructProp
from .transport import sinkhorn

# 融合权重网格。**CUSSM 与 LinFuse（以及任何做两视图线性融合的对照）必须共用同一张**，
# 否则比较的是网格分辨率而非方法差异。实测：α 在 0.05 处有尖峰，21 点网格能命中而
# 11 点网格不能，仅此一项就造成 0.04 的 Hits@1 落差。
ALPHA_GRID = tuple(round(float(x), 3) for x in np.linspace(0.0, 1.0, 21))


# ---------------------------------------------------------------- 对齐原语
def ridge(X: np.ndarray, Y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """岭回归闭式解 W = (XᵀX + λI)⁻¹XᵀY。"""
    d = X.shape[1]
    A = X.T.astype(np.float64) @ X.astype(np.float64) + lam * np.eye(d)
    B = X.T.astype(np.float64) @ Y.astype(np.float64)
    return np.linalg.solve(A, B).astype(np.float32)


def ortho_procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """正交 Procrustes：min‖XW−Y‖_F s.t. WᵀW=I（SVD 闭式解）。

    本项目实测：在语义视图上 **Procrustes 明显优于岭回归**
    （fr_en 全候选集 Hits@1：0.6075 vs 0.3250）。岭回归有 d² 个自由参数，
    而种子只有 4,050 条，严重过拟合；正交约束把参数降到 d(d−1)/2 且是保内积的。
    """
    U, _s, Vt = np.linalg.svd(X.T.astype(np.float64) @ Y.astype(np.float64))
    return (U @ Vt).astype(np.float32)


def _rownorm(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (M / n).astype(np.float32)


def _zscore(M: np.ndarray) -> np.ndarray:
    """逐行 z-score：把不同路由的得分放到同一尺度，再加权融合。

    必须做这一步 —— 余弦得分与"属性相似度"量纲不同，直接线性加权等于让量纲大的
    那一路独占权重（实测：不做标准化时 α 的网格最优值恒为端点）。
    """
    m = M.mean(axis=1, keepdims=True)
    s = M.std(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return ((M - m) / s).astype(np.float32)


# ---------------------------------------------------------------- 统一类型骨架
# `π_A, π_B` 是**数据侧**的约定（4.1 定义 4），与具体方法无关：它由两侧**结构特征**
# 合并聚类得到，只依赖图本身与 (d_sem, n_types, seed)，**不依赖训练种子/划分**。
# 故可跨方法共享 —— 这正是"基线的 ρ_fib 与本文方法同一把尺子"的技术含义。
# 缓存键含图身份与超参，命中即复用（同一数据集上 9 个方法只算一次 k-means）。
# 注意：**不能**把它塞进 `pair.meta` —— `run_experiment` 会把 `pair.meta` 整份写进
# 产物 JSON（`rec["meta"] = pair.meta`），ndarray 会直接让 `json.dump` 抛错。
_FIBER_CACHE: dict = {}


def _fiber_labels_of(pair: "Pair", d_sem: int = 256, n_types: int = 16,
                     seed: int = 2026):
    """取统一类型骨架 `(lab_L, lab_R)`；同一图与超参下只算一次。"""
    key = (pair.left.name, pair.left.n, pair.right.name, pair.right.n,
           int(d_sem), int(n_types), int(seed))
    lab = _FIBER_CACHE.get(key)
    if lab is None:
        VL = build_view(pair.left, 64, d_sem, seed)
        VR = build_view(pair.right, 64, d_sem, seed)
        lab = fibers_from(VL.struct, VR.struct, n_types, seed=seed)
        _FIBER_CACHE[key] = lab
    return lab


# ---------------------------------------------------------------- 基类
class BaseMatcher:
    """所有方法与消融的统一接口（第 5.4.4 节契约）。"""
    name = "base"
    family = "?"
    uses_fiber = False

    def fit(self, pair: Pair) -> "BaseMatcher":
        raise NotImplementedError

    def score_block(self, pair: Pair, left_idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def routes(self, pair: Pair, left_idx: np.ndarray) -> dict:
        """各路由此方法提供的原始得分；单路由方法只返回一个键。"""
        return {"主路由": self.score_block(pair, left_idx)}

    def fuse(self, parts: dict) -> np.ndarray:
        """路由的融合方式 —— Δ_glue 度量的正是它与"逐块并置"之间的偏移。"""
        return np.mean(np.stack(list(parts.values())), axis=0)

    def fusion_routes(self) -> list[str] | None:
        """**真正进入融合**的路由名；None 表示"`routes()` 给的全进"。

        审计要用它来定"逐块并置"的基准：若一条路由被提供但权重为零，
        它不该出现在基准里，否则模型会因为"备了一条没用上的路由"而被扣分。
        （实测缺陷：δ=0 时 CUSSM 的 Δ_glue 被算成 0.6448 而 LinFuse 只有 0.2292，
        差异**全部**来自那条权重为零的粘合路由。）
        """
        return None

    def n_params(self) -> int:
        return 0

    def n_embed_params(self) -> int:
        return 0

    def param_detail(self) -> dict:
        return {}

    def fiber_labels(self):
        """统一类型骨架 `π_A, π_B` —— **数据侧**约定（4.1 定义 4），与具体方法无关。

        纤维保持率 `ρ_fib = (1/N) Σ 1{π_B(F(x_i)) = π_A(x_i)}`（4.3.3）只要求方法给出
        预测 `F(x_i)`，故它**对任何方法都成立**，不是本文方法专属指标。实现原本只在
        `CUSSM` 里返回标签、基类一律返回 `None` ⇒ 表 12 该列对 8 个基线全为空。此处改为
        基类统一提供：`fit()` 之后由 `bind_fibers()` 绑定。

        未绑定时**如实返回 `None`** —— 不以零向量冒充标签（零标签会让 `ρ_fib` 恒为 1，
        是比"缺数"严重得多的错误）。
        """
        return getattr(self, "_fib", None)

    def bind_fibers(self, pair: "Pair", d_sem: int = 256, n_types: int = 16,
                    seed: int = 2026) -> "BaseMatcher":
        """绑定统一类型骨架，使各行 `ρ_fib` 与本文方法**同一把尺子**。

        参数须与本次运行一致（`run_experiment` 传入 `args.d_sem/n_types/seed`）；因为走
        的是与 `CUSSM.fit` **同一个** `_fiber_labels_of` 缓存，与 CUSSM 的 `lab_L/lab_R`
        **逐位相同**（`run_experiment` 内有断言核对，不一致即报错，不静默继续）。
        """
        self._fib = _fiber_labels_of(pair, d_sem=d_sem, n_types=n_types, seed=seed)
        return self

    def flags(self) -> dict:
        return {"fiber": bool(self.uses_fiber)}


# ---------------------------------------------------------------- CUSSM
class CUSSM(BaseMatcher):
    name = "CUSSM（本文方法）"
    family = "本文方法"

    def __init__(self, rounds: int = 0, rounds_grid=(0, 1, 2),
                 conf_q: float = 55.0, max_new: int = 3000,
                 d_sem: int = 256, n_types: int = 16, seed: int = 2026,
                 prop_chunk: int = 512, use_route_H: bool = True,
                 use_route_K: bool = True, use_fiber: bool = True,
                 use_type: bool = True, use_glue: bool = True, tune: bool = True,
                 grid_alpha=ALPHA_GRID,
                 grid_beta=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
                 grid_gamma=(0.0, 0.05, 0.15, 0.4),
                 grid_delta=(0.0, 0.05, 0.15, 0.3),
                 grid_w=(0.0, 0.25, 0.5, 0.75, 1.0),
                 topk_sl: int = 20, sinkhorn_eps: float = 0.05,
                 sinkhorn_iter: int = 60, min_gain: float = 0.004,
                 z_margin: float = 2.0, channels_must_on: bool = False, **kw):
        self.rounds, self.rounds_grid = rounds, rounds_grid
        self.conf_q, self.max_new = conf_q, max_new
        self.topk_sl, self.sinkhorn_eps, self.sinkhorn_iter = (
            topk_sl, sinkhorn_eps, sinkhorn_iter)
        self.min_gain = min_gain
        self.z_margin = z_margin
        # 「通道强制开启」消融开关（定义见 run_experiment.py 的 FORCE_ABLATIONS）。
        # True 时**取消 M2/M3 的显著性门槛**，改在**非零**网格内取验证集 argmax，
        # 从而保证 β、γ、δ 均 >0。动机：门槛回答的是"该不该采纳这条通道"，
        # 而消融要回答的是"通道启用后能带来多少增益" —— 两者不是同一个问题，
        # 故度量增益时必须绕开门槛（否则在门槛判"不采纳"的数据集上无从度量）。
        # 默认 False ⇒ 主表与既有消融的行为逐位不变。
        self.channels_must_on = bool(channels_must_on)
        self.tune_log: list = []
        self.d_sem, self.n_types, self.seed = d_sem, n_types, seed
        self.prop_chunk = prop_chunk
        self.use_route_H, self.use_route_K = use_route_H, use_route_K
        self.use_fiber, self.use_type, self.use_glue = use_fiber, use_type, use_glue
        self.tune = tune
        self.grid_alpha, self.grid_beta = grid_alpha, grid_beta
        self.grid_gamma, self.grid_delta = grid_gamma, grid_delta
        self.grid_w = grid_w
        self.alpha, self.beta, self.gamma, self.delta = 0.5, 0.0, 0.0, 0.0
        self.w = 0.5
        self.rounds_selected = rounds
        self.uses_fiber = use_fiber
        self.train_time = 0.0
        self.param_detail_: dict = {}
        self.SP = None

    # ---------------- 拟合 ----------------
    def fit(self, pair: Pair) -> "CUSSM":
        t0 = time.time()
        if not (self.use_route_H or self.use_route_K):
            raise ValueError("两条路由至少要启用一条")
        sl, sr = pair.seeds[:, 0], pair.seeds[:, 1]

        # ---- 结构路由 H：种子锚定的跨图结构画像（可选迭代传播）----
        # **传播轮数在验证种子上选，不看测试集**。实测（`.workbuddy/diag_prop.py`）：
        # 迭代自训练在 DBP15K 上**反而降低精度**（fr_en 全候选集 Hits@1：
        # 0 轮 0.582 → 1 轮 0.454 → 3 轮 0.422）—— 互近邻置信度不足以支撑自我强化，
        # 错误锚点会沿边传播污染画像。验证集上的走势与测试集一致，故"选 0 轮"
        # 是数据驱动的结论，不是事后挑最好看的数。
        # **关掉结构路由时不必构造它** —— 传播是全流程最贵的一步，
        # 消融"CUSSM − 结构路由H"若照旧构造，等于让被消融的组件仍然付费。
        self.SP, self.rounds_selected = (
            self._fit_struct(pair) if self.use_route_H else (None, 0))

        # ---- 语义路由 K：表层名块与属性值块**分别**对齐后加权合成 ----
        # 两块必须分开对齐、分开归一化：合并后统一归一化会互相稀释
        # （实测 fr_en 全候选集 Hits@1：合并 0.635，名字单独 0.762，属性单独 0.650）。
        # 配比 w 由验证集选出。
        self.VL = build_view(pair.left, 64, self.d_sem, self.seed)
        self.VR = build_view(pair.right, 64, self.d_sem, self.seed)
        self.W_n = ortho_procrustes(self.VL.sem_name[sl], self.VR.sem_name[sr])
        self.W_v = ortho_procrustes(self.VL.sem_val[sl], self.VR.sem_val[sr])
        self.NL = _rownorm(self.VL.sem_name @ self.W_n)
        self.NR = self.VR.sem_name
        self.AL_ = _rownorm(self.VL.sem_val @ self.W_v)
        self.AR_ = self.VR.sem_val
        self.w = 0.5                      # 字面权重，_tune 里定

        # ---- 纤维（打字投影）：两侧的"结构角色"特征合并聚类 → 共享类型空间 ----
        # ---- 纤维（打字投影）：两侧的"结构角色"特征合并聚类 → 共享类型空间 ----
        if self.use_fiber or self.use_type:
            # 走与基类 `bind_fibers` **同一个**缓存 ⇒ 基线与本文方法的类型骨架是
            # 同一套对象（逐位相同），且 9 个方法只付一次 k-means 的代价。
            self.lab_L, self.lab_R = _fiber_labels_of(pair, d_sem=self.d_sem,
                                                      n_types=self.n_types,
                                                      seed=self.seed)
        else:
            self.lab_L = np.zeros(self.VL.n, np.int32)
            self.lab_R = np.zeros(self.VR.n, np.int32)
        self.n_lab = int(max(self.lab_L.max(), self.lab_R.max())) + 1

        # ---- 类型先验 P(右侧纤维 | 左侧纤维)：由种子统计 ----
        T = np.zeros((self.n_lab, self.n_lab), np.float32)
        np.add.at(T, (self.lab_L[sl], self.lab_R[sr]), 1.0)
        rs = T.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        self.type_prior = (T / rs).astype(np.float32)

        self.param_detail_ = {
            "语义路由映射 W_字面": int(self.d_sem * self.d_sem),
            "语义路由映射 W_属性": int(self.d_sem * self.d_sem),
            "类型先验表": int(self.type_prior.size),
            "融合标量 w,α,β,γ,δ": 5,
            "结构传播新增匹配对数": (0 if self.SP is None
                                     else int(self.SP.nM - len(pair.seeds))),
        }
        if self.tune:
            self._tune(pair)
        self.train_time = time.time() - t0
        return self

    # ---------------- 结构路由的轮数选择 ----------------
    def _fit_struct(self, pair: Pair):
        """按验证种子上的全候选集 Hits@1 选传播轮数（默认 {0,1,2}）。"""
        grid = (self.rounds,) if not self.rounds_grid else tuple(self.rounds_grid)
        best, best_h = None, -1.0
        for r in grid:
            sp = StructProp(rounds=r, conf_q=self.conf_q, max_new=self.max_new,
                            chunk=self.prop_chunk, seed=self.seed).fit(pair)
            h = self._val_hits(sp, pair)
            if h > best_h + 1e-12:
                best, best_h = (sp, r), h
        return best[0], best[1]

    @staticmethod
    def _val_hits(sp: StructProp, pair: Pair) -> float:
        vi, vj = pair.val[:, 0], pair.val[:, 1]
        hit = 0
        for i in range(0, len(vi), 512):
            blk = vi[i:i + 512]
            S = sp.score_block(blk)
            hit += int((np.argmax(S, axis=1) == vj[i:i + 512]).sum())
            del S
        return hit / max(len(vi), 1)

    # ---------------- 各路得分 ----------------
    def _route_H(self, idx: np.ndarray) -> np.ndarray:
        return self.SP.score_block(idx)

    def _route_K(self, idx: np.ndarray) -> np.ndarray:
        """语义路由：字面视图与属性视图的余弦，按字面权重 $w$ 加权。

        两次 `(m,d) @ (d,N)` 矩阵乘经 `dev.mix_matmul` 合并为**一次**设备往返。
        以 fr_en 为例（d=512、N=105,889、chunk=512）：单块的乘加量约 5.6×10^10 FLOP，
        CPU 上约秒级，T4 上是十毫秒级——这是本方法最值得上 GPU 的一处。
        数值与"分别两次矩阵乘再加权"逐位一致（float32 累加顺序不变）。
        """
        return dev.get().mix_matmul(self.NL[idx], self.NR.T, self.w,
                                    self.AL_[idx], self.AR_.T, 1.0 - self.w)

    def _k_parts(self, idx: np.ndarray):
        """语义路由的两个子通道的**原始读数**（未加权）。"""
        return ((self.NL[idx] @ self.NR.T).astype(np.float32),
                (self.AL_[idx] @ self.AR_.T).astype(np.float32))

    def _route_C_shortlist(self, idx: np.ndarray):
        """粘合路由的**短名单与百分位读数**（`_route_C` 的分块内核）。

        与 `_route_C` 的差别只有一处：`pool` 不再整体物化 —— 它按行块算出后
        只留下**短名单位置上的值**（`(m, ≤3k)` 的小矩阵）。sinkhorn 与百分位
        标准化本来就只在这个小矩阵上做（`keep` 列），故返回的
        `(S, zc, floor)` 与原实现逐位相同。

        为什么必须这样做（2026-09-29）：原实现对全量 `val` 物化
        `H / K / zH / zK / pool / out` 共 6 个 `(n_val, N_R)` 矩阵。在 Track A′
        的规模上（候选池 ≈1.9×10^5、n_val≈5.4×10^3）约 25 GB，是 `dmesg`
        所记 anon-rss 28.5 GB 那次 OOM 的直接来源；yago3_10 上亦有 9.8 GB。

        跨行依赖只有三处，且都**不在** `pool` 上，故分块安全：
          * `keep = max_row(非 -1 列数)`  —— 全量 `(m, ≤3k)` 的索引矩阵，很小；
          * sinkhorn 的列归一化          —— 在 `(m, keep)` 上解，keep 为百量级；
          * `floor = zc.min() - 1`        —— 全量 `zc` 本身只有 `(m, keep)`。
        """
        m = len(idx)
        k = self.topk_sl
        both = self.use_route_H and self.use_route_K
        ncol = 3 * k if both else k
        CH = mem.plan_block(self.VR.n, arrays=6, chunk=self.prop_chunk)

        UNIQ = np.full((m, ncol), -1, np.int64)
        CM = np.zeros((m, ncol), np.float32)
        LABC = np.zeros((m, ncol), np.int32)
        for sl in mem.block_slices(m, CH):
            zb = idx[sl]
            H = self._route_H(zb) if self.use_route_H else None
            K = self._route_K(zb) if self.use_route_K else None
            if H is None and K is None:
                return None, None, None
            zH = _zscore(H) if H is not None else None
            zK = _zscore(K) if K is not None else None
            pool = (zH + zK) / 2.0 if (zH is not None and zK is not None) \
                else (zH if zH is not None else zK)

            nb = pool.shape[1]
            cand = np.argpartition(-pool, min(k, nb - 1), axis=1)[:, :k]
            if zH is not None and zK is not None:
                cand = np.hstack([cand,
                                  np.argpartition(-zH, min(k, nb - 1),
                                                  axis=1)[:, :k],
                                  np.argpartition(-zK, min(k, nb - 1),
                                                  axis=1)[:, :k]])
            # 每行去重（保持列数一致：用 -1 填充）——逐行独立，与整块同序
            uniq = UNIQ[sl]
            for i in range(len(zb)):
                seen, w = set(), 0
                for c in cand[i]:
                    if c not in seen:
                        seen.add(c)
                        uniq[i, w] = c
                        w += 1
            # 只留短名单位置上的 pool 值（与整块实现同为 pool[arange, S]）
            CM[sl] = -pool[np.arange(len(zb))[:, None], uniq]
            if self.use_fiber:
                LABC[sl] = self.lab_R[uniq]
            del H, K, zH, zK, pool, cand

        keep = int((UNIQ >= 0).sum(axis=1).max())
        S = UNIQ[:, :keep]
        Cm = CM[:, :keep]
        mask = None
        if self.use_fiber:
            mask = (self.lab_L[idx][:, None] != LABC[:, :keep])
            # 整行全禁时放开（否则该行传输问题无可行解）
            bad = mask.all(axis=1)
            if bad.any():
                mask[bad] = False
        P = sinkhorn(Cm, eps=self.sinkhorn_eps, mask=mask,
                     n_iter=self.sinkhorn_iter)
        logP = np.log(np.maximum(P, 1e-30)).astype(np.float32)

        keep_n = S.shape[1]
        order = np.argsort(-logP, axis=1, kind="stable")
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(keep_n)[None, :], axis=1)
        pct = 1.0 - (rank.astype(np.float32) + 0.5) / float(keep_n)
        zc = (pct - pct.mean(axis=1, keepdims=True)) / (pct.std(axis=1, keepdims=True)
                                                         + np.float32(1e-9))
        floor = float(zc.min()) - 1.0
        return S, zc, floor
    def _route_C(self, idx: np.ndarray) -> np.ndarray:
        """粘合路由（余极限粘合的算法落地）：候选短名单上的**全局一致性传输**。

        做法（4.4 的 "Pool 之后作商" 的可操作形态）：

          ① 把两路由的标准化读数并置，得 Pool = (zH + zK)/2（与 α 无关，
             故粘合读数不随融合权重漂移）；
          ② 取每个查询在 H、K 上各自的 top-`self.topk_sl` 并集，构成短名单
             （≤ 2·topk_sl 个候选）—— 这一步是必要的：完整候选有 10⁵ 量级，
             物化 n×N 的传输问题需要数十 GB；
          ③ 在"本批查询 × 短名单"的小矩阵上解熵正则最优传输（Sinkhorn），
             并把**纤维硬约束作为掩码**在传输前施加（而不是解完再过滤）——
             这正是 4.2 的"约束在最优点内"的落点；
          ④ 取传输计划 log P 在短名单**内的序**（映射为百分位后逐行标准化）作为
             第三路读数：它奖励那些在**全局**（整批查询共享同一批目标）也站得住的
             配对，而不只是单行 argmax 的配对。取"序"而非 log P 本身是为尺度安全 ——
             详见图下注释（直接 z-score log P 会让 δ 一开就崩）。

        短名单之外的候选赋一个低于短名单最低分的常数，故该路由只在短名单上
        产生排序信息，不改变短名单之外的相对次序。
        """
        S, zc, floor = self._route_C_shortlist(idx)
        m = len(idx)
        out = np.full((m, self.VR.n), np.float32(-1e4))
        out[np.arange(m)[:, None], S] = zc
        out[out < -1e3] = floor
        # 与原实现同为 float32；不再 `.astype(np.float32)` —— 那是同 dtype 的
        # **整份复制**，在 Track A′ 的规模上白占约 4 GB 峰值（数值完全相同）。
        return out


    def routes(self, pair: Pair, left_idx: np.ndarray) -> dict:
        out = {}
        H = self._route_H(left_idx) if self.use_route_H else None
        K = self._route_K(left_idx) if self.use_route_K else None
        if H is not None:
            out["H 结构路由"] = H
        if K is not None:
            out["K 语义路由"] = K
        if self.use_glue and H is not None and K is not None:
            out["C 粘合路由"] = self._route_C(left_idx)
        return out

    def fusion_routes(self):
        """只有 α 与 1−α 都非零时两条路由才都进融合。"""
        out = []
        if self.alpha > 0:
            out.append("H 结构路由")
        if self.alpha < 1:
            out.append("K 语义路由")
        if self.delta > 0:
            out.append("C 粘合路由")
        return out

    def fuse(self, parts: dict) -> np.ndarray:
        """z-score 标准化后加权融合 —— 量纲不同，必须先标准化。"""
        H, K, C = parts.get("H 结构路由"), parts.get("K 语义路由"), parts.get("C 粘合路由")
        if H is None:
            base = K
        elif K is None:
            base = H
        else:
            base = self.alpha * _zscore(H) + (1.0 - self.alpha) * _zscore(K)
        if C is not None and self.delta:
            base = base + np.float32(self.delta) * _zscore(C)
        return base.astype(np.float32)

    def _penalty(self, idx: np.ndarray) -> np.ndarray:
        """纤维罚 + 类型先验（先验是加分，故取正号）。"""
        P = np.zeros((len(idx), self.VR.n), np.float32)
        if self.use_fiber and self.beta:
            P -= np.float32(self.beta) * (self.lab_L[idx][:, None] != self.lab_R[None, :])
        if self.use_type and self.gamma:
            P += np.float32(self.gamma) * self.type_prior[self.lab_L[idx]][:, self.lab_R]
        return P

    def score_block(self, pair: Pair, left_idx: np.ndarray) -> np.ndarray:
        return self.fuse(self.routes(pair, left_idx)) + self._penalty(left_idx)

    # ---------------- 超参选择（验证种子的**完整候选集**上做分阶段搜索）----------------
    @staticmethod
    def _se(n: int, p: float) -> float:
        """验证集 Hits@1 的二项标准误（p 截断到 [1e-6, 1-1e-6] 防除零/零方差）。"""
        p = min(max(float(p), 1e-6), 1.0 - 1e-6)
        return float(np.sqrt(p * (1.0 - p) / max(int(n), 1)))

    def _need(self, n: int, h: float) -> float:
        """**噪声门槛**：附加通道的验证集增益须超过 `z_margin` 个标准误才采纳。

        为什么不能用固定常数（原实现取 min_gain=0.004）：0.004 与验证集规模无关，
        在 n=450 时远小于其标准误（≈0.014），于是调参器会把噪声当信号。
        实测（DBP15K fr_en）：CUSSM 在 α=0（结构路由关闭，与 LinFuse 同构）下仍被
        选中 β=0.25 的纤维罚，验证集 +0.004 以上、**测试集 −0.053**，最终 0.8653
        < LinFuse 0.9180 —— 这是纯粹的自伤。改用 2 个标准误（≈0.026）后该通道被拒。
        """
        return float(max(self.z_margin * self._se(n, h), self.min_gain))

    def _tune(self, pair: Pair) -> None:
        """在验证种子上、**完整候选集**内按「前向选择 + 2-SE 停止规则」选超参。

        模型族按复杂度递增，逐级**付费升级**：

            M1  两条路由融合的 α            （**共享部件**：与 LinFuse 同规则，无门槛）
            M2  + 粘合通道 C，δ 自由         （CUSSM 专属，须过 2-SE 门槛）
            M3  + 纤维罚 β 与类型先验 γ      （CUSSM 专属，须过 2-SE 门槛）

        **门槛只加在 CUSSM 专属通道上**：α 与阶段 1 的 w 是 CUSSM 与 LinFuse 共享的
        部件，必须用同一张网格、同一条自由 argmax 规则，否则比较的就变成选参规则。
        由此保证 `CUSSM` 在 δ=β=γ=0 时**逐位等于** `LinFuse`，附加通道只可能加分。

        为什么不在"抽样干扰项"上选参：实测 1,000 路抽样候选会把 α 顶到 1.0
        （抽样候选里"名字几乎全同"的难例被稀释），代理指标与真实指标**排序不一致**。
        """
        vi, vj = pair.val[:, 0], pair.val[:, 1]
        NR = self.VR.n
        n_v = int(len(vi))
        # ================= 按行分块（2026-09-29 消除 OOM）=================
        # 原实现在**全量 val** 上同时持有 H / zH / AN / AV / zK / pen_f / pen_t /
        # zC / base / S 共 8 组 (n_val, N_R) 矩阵：yago3_10（n_val=3218、N=109,398）
        # 算术和 43.5 GB、Track A′（候选池 ≈1.9e5）119.9 GB —— 两条链都被内核 OOM
        # 杀死（dmesg：anon-rss 27.9 GB / 28.5 GB）。
        #
        # 改为**外块内网格**：每块算齐本轮需要的读数 → 在该块内跑完整张网格 →
        # 只把**整数**命中数累加到全局。峰值 = chunk × N_R × itemsize × 块内活跃
        # 数组数，与 n_val 解耦（可用 CUSSM_BLOCK_BUDGET_MB 进一步收紧）。
        #
        # 数值等价的三项前提，逐条成立且已断言（`.workbuddy/ab_chunk_parity.py`）：
        #   ① `_zscore` 走 axis=1，逐行独立；
        #   ② `np.argmax(..., axis=1)` 只看该行；
        #   ③ 命中数用整数累加、**只在最后除一次**（逐块 `.mean()` 再相加会引入
        #      与整块不同的浮点舍入）。
        CH = mem.plan_block(NR, arrays=8, chunk=self.prop_chunk)

        def _parts(sl, need_pen: bool):
            """块级读数 zH / zK / pen_f / pen_t（未启用的通道给零阵，保持原语义）。"""
            zb = vi[sl]
            H = self._route_H(zb) if self.use_route_H else None
            zHb = _zscore(H) if H is not None else None
            zKb = _zscore(self._route_K(zb)) if self.use_route_K else None
            if not need_pen:
                return zHb, zKb, None, None
            pf = (np.zeros((len(zb), NR), np.float32) if not self.use_fiber else
                  -(self.lab_L[zb][:, None] != self.lab_R[None, :]).astype(np.float32))
            pt = (np.zeros((len(zb), NR), np.float32) if not self.use_type else
                  self.type_prior[self.lab_L[zb]][:, self.lab_R])
            return zHb, zKb, pf, pt

        # ---- 阶段 1：语义路由内部配比 w（名字 vs 属性）----
        # 这两块是**同一条**语义路由的两个子通道，w 的搜索口径与 LinFuse 完全一致
        # （含不加门槛），以保证"关闭全部附加通道时 CUSSM ≡ LinFuse"。
        self.w = 0.5
        self.w_hits1 = float("nan")
        if self.use_route_K:
            acc_w = [0] * len(self.grid_w)
            for sl in mem.block_slices(n_v, CH):
                ANb, AVb = self._k_parts(vi[sl])
                gb = vj[sl]
                for i, w in enumerate(self.grid_w):
                    S = np.float32(w) * ANb + np.float32(1.0 - w) * AVb
                    acc_w[i] += int((np.argmax(S, axis=1) == gb).sum())
                del ANb, AVb
            bw, bh = 0.5, -1.0
            for i, w in enumerate(self.grid_w):
                h = acc_w[i] / max(n_v, 1)
                if h > bh + 1e-12:
                    bw, bh = float(w), h
            self.w, self.w_hits1 = bw, bh

        log = []
        # ---- M0（仅语义路由，α=0）与 M1（α 自由）在同一遍里取 ----
        acc_a = [0] * len(self.grid_alpha)
        acc_h0 = 0
        for sl in mem.block_slices(n_v, CH):
            zHb, zKb, _pf, _pt = _parts(sl, need_pen=False)
            gb = vj[sl]
            if zKb is not None:
                acc_h0 += int((np.argmax(zKb, axis=1) == gb).sum())
            elif zHb is not None:
                acc_h0 += int((np.argmax(zHb, axis=1) == gb).sum())
            if zHb is not None and zKb is not None:
                for i, a in enumerate(self.grid_alpha):
                    acc_a[i] += int(
                        (np.argmax(a * zHb + (1.0 - a) * zKb, axis=1) == gb).sum())
            del zHb, zKb
        h0 = acc_h0 / max(n_v, 1)
        self.alpha = 0.0 if self.use_route_K else 1.0
        self.delta = 0.0
        self.beta = self.gamma = 0.0
        cur = h0
        # **α 不加门槛** —— α（与阶段 1 的 w）是 CUSSM 与 LinFuse **共享**的部件，
        # 必须用完全相同的选择规则（自由 argmax、同一张 ALPHA_GRID），否则比较的
        # 是选参规则而不是方法。若在此加门槛，CUSSM 会在"增益真实但小于门槛"时
        # 停在 α=0 而 LinFuse 照取 α>0，CUSSM 反而落后 —— 本轮实测就发生过一次。
        # 门槛只用于 CUSSM **专属**的附加通道（M2 粘合、M3 纤维/类型）。
        if self.use_route_H and self.use_route_K:
            b = (h0, 0.0)
            for i, a in enumerate(self.grid_alpha):
                h = acc_a[i] / max(n_v, 1)
                if h > b[0] + 1e-12:
                    b = (h, float(a))
            log.append(("M1 结构路由", h0, b[0], 0.0, b[1] != 0.0))
            self.alpha, cur = b[1], b[0]

        # ---- M2：粘合通道（δ 自由）----
        # `zC` 只保留一次全量出口：粘合读数按定义就是 (n_val, N_R) 的稠密场，
        # 而其**内部**的全量物化已由 `_route_C_shortlist` 消除。
        zC = (self._route_C(vi) if (self.use_glue and self.use_route_H
                                    and self.use_route_K) else None)
        if zC is not None:
            acc_d = [0] * len(self.grid_delta)
            for sl in mem.block_slices(n_v, CH):
                zHb, zKb, _pf, _pt = _parts(sl, need_pen=False)
                gb = vj[sl]
                S0 = self.alpha * zHb + (1.0 - self.alpha) * zKb
                zCz = _zscore(zC[sl])
                for i, dl in enumerate(self.grid_delta):
                    if dl == 0.0:
                        continue
                    acc_d[i] += int(
                        (np.argmax(S0 + np.float32(dl) * zCz, axis=1) == gb).sum())
                del zHb, zKb, S0, zCz
            if self.channels_must_on:
                h0m = cur
                bb = None                # 非零网格内的验证集 argmax，不论是否优于基线
                for i, dl in enumerate(self.grid_delta):
                    if dl == 0.0:
                        continue
                    h = acc_d[i] / max(n_v, 1)
                    if bb is None or h > bb[0] + 1e-12:
                        bb = (h, float(dl))
                if bb is not None:
                    self.delta, cur = bb[1], bb[0]
                    log.append(("M2 粘合通道[强制]", h0m, bb[0], 0.0, True))
            else:
                b = (cur, 0.0)
                for i, dl in enumerate(self.grid_delta):
                    if dl == 0.0:
                        continue
                    h = acc_d[i] / max(n_v, 1)
                    if h > b[0] + 1e-12:
                        b = (h, float(dl))
                thr = self._need(n_v, b[0])
                ok = b[0] - cur >= thr
                log.append(("M2 粘合通道", cur, b[0], thr, ok))
                if ok:
                    self.delta, cur = b[1], b[0]

        # ---- M3：纤维罚与类型先验（β、γ）----
        # 先把待评估的 (β, γ) 组合定下来（含强制模式下的跳过规则），再分块累加。
        grid_b = tuple(self.grid_beta) if self.use_fiber else (0.0,)
        grid_g = tuple(self.grid_gamma) if self.use_type else (0.0,)
        pairs_bg = []
        for bbv in grid_b:
            for gv in grid_g:
                if bbv == 0.0 and gv == 0.0:
                    continue
                # 强制模式下，**已启用**的那一路系数必须取非零值 —— 否则
                # "强制开启"会退化成"仍被选成 0"。取消启用的那一路按设计保持 0。
                if self.channels_must_on and ((self.use_fiber and bbv == 0.0)
                                              or (self.use_type and gv == 0.0)):
                    continue
                pairs_bg.append((float(bbv), float(gv)))
        b2 = (cur, 0.0, 0.0)
        b2n = None                # 非零 (β,γ) 网格内的 argmax，不论是否优于基线
        if pairs_bg:
            acc_bg = [0] * len(pairs_bg)
            for sl in mem.block_slices(n_v, CH):
                zHb, zKb, pf, pt = _parts(sl, need_pen=True)
                gb = vj[sl]
                if zHb is not None and zKb is not None:
                    base = self.alpha * zHb + (1.0 - self.alpha) * zKb
                else:
                    base = zHb if zKb is None else zKb
                if self.delta:
                    base = base + np.float32(self.delta) * _zscore(zC[sl])
                for i, (bbv, gv) in enumerate(pairs_bg):
                    S = base + np.float32(bbv) * pf + np.float32(gv) * pt
                    acc_bg[i] += int((np.argmax(S, axis=1) == gb).sum())
                del zHb, zKb, pf, pt, base
            for i, (bbv, gv) in enumerate(pairs_bg):
                h = acc_bg[i] / max(n_v, 1)
                if h > b2[0] + 1e-12:
                    b2 = (h, bbv, gv)
                if b2n is None or h > b2n[0] + 1e-12:
                    b2n = (h, bbv, gv)
        thr = self._need(n_v, b2[0])
        if self.channels_must_on:
            # 取消门槛：只要网格里存在非零 (β,γ) 就强制采其 argmax。
            ok = b2n is not None
            sel = b2n if ok else b2
            log.append(("M3 纤维/类型[强制]", cur, sel[0], 0.0, ok))
        else:
            ok = (b2[1] > 0.0 or b2[2] > 0.0) and (b2[0] - cur >= thr)
            sel = b2
            log.append(("M3 纤维/类型", cur, b2[0], thr, ok))
        if ok:
            self.beta, self.gamma, cur = sel[1], sel[2], sel[0]

        # ---- 细化：通道集合已定，对 α 再走一轮坐标上升（只在粘合通道被采纳时才发生，
        #      故未采纳任何附加通道时 α 与 LinFuse 的选择逐位相同）----
        if self.delta:
            acc_a2 = [0] * len(self.grid_alpha)
            for sl in mem.block_slices(n_v, CH):
                zHb, zKb, pf, pt = _parts(sl, need_pen=True)
                gb = vj[sl]
                zCz = _zscore(zC[sl])
                for i, a in enumerate(self.grid_alpha):
                    if zHb is not None and zKb is not None:
                        S = a * zHb + (1.0 - a) * zKb
                    else:
                        S = zHb if zKb is None else zKb
                    S = (S + np.float32(self.delta) * zCz
                         + np.float32(self.beta) * pf + np.float32(self.gamma) * pt)
                    acc_a2[i] += int((np.argmax(S, axis=1) == gb).sum())
                del zHb, zKb, pf, pt, zCz
            b = (cur, self.alpha)
            for i, a in enumerate(self.grid_alpha):
                h = acc_a2[i] / max(n_v, 1)
                if h > b[0] + 1e-12:
                    b = (h, float(a))
            self.alpha, cur = b[1], b[0]

        self.val_hits1 = float(cur)
        self.tune_log = log
        self.tune_proxy = ("验证种子 + 完整候选集；前向选择，附加通道须增益 ≥ "
                           f"{self.z_margin}×二项标准误（n_val={n_v}）")


    # ---------------- 统计 ----------------
    def fiber_labels(self):
        """即便本次消融未启用纤维约束，也返回同一套标签，以便公平观察纤维保持率。"""
        return (self.lab_L, self.lab_R)

    def n_params(self) -> int:
        """**可训练映射参数**（不含随数据集规模线性增长的查表参数）。"""
        return int(sum(v for k, v in self.param_detail_.items()
                       if "匹配对数" not in k))

    def n_embed_params(self) -> int:
        return 0

    def param_detail(self) -> dict:
        return dict(self.param_detail_)

    def flags(self) -> dict:
        return {"fiber": bool(self.use_fiber), "type": bool(self.use_type),
                "glue": bool(self.use_glue), "route_H": bool(self.use_route_H),
                "route_K": bool(self.use_route_K),
                "传播轮数(验证集选出)": int(self.rounds_selected),
                "传播新增匹配对数": int(0 if self.SP is None
                                       else self.SP.nM - self.SP.n_seed),
                "α": round(float(self.alpha), 3), "β": round(float(self.beta), 3),
                "γ": round(float(self.gamma), 3), "δ": round(float(self.delta), 3),
                "w 字面权重": round(float(self.w), 3),
                "val hits@1": round(float(getattr(self, "val_hits1", float("nan"))), 4),
                # 准入判据是 `增益 ≥ 阈值`（见 tune_proxy 与 ok = b[0]-cur >= thr），
                # 故比较符必须随判定结果取 `≥`/`<`。原实现把 `<` 写死，导致**通过**准入的
                # M1 被印成「✓(+0.0400<0.0000)」——读起来是假命题、与前面的 ✓ 自相矛盾，
                # 且恰好把「M2/M3 为何系数取 0」的证据表述得含混。仅改显示，不动任何数值。
                "通道准入": [f"{s}:{'✓' if ok else '✗'}"
                             f"({h1 - h0:+.4f}{'≥' if ok else '<'}{thr:.4f})"
                             for s, h0, h1, thr, ok in getattr(self, "tune_log", [])]}
