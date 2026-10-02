# -*- coding: utf-8 -*-
"""结构路由 H 的实现：**种子锚定的跨图迭代结构传播**（免训练）。

## 为什么不用 KG 嵌入 + Procrustes / 联合训练

本项目 2026-09-27 在 DBP15K fr_en 上逐层实测（见 `.workbuddy/diag_*.py`）：

| 路线 | 全候选集 Hits@1（105,889 候选） |
|---|---|
| 两侧独立训练 TransE + 事后正交 Procrustes | 0.005 |
| MTransE 式联合训练（平方误差对齐损失） | 0.000（种子余弦 0.997 —— 角度塌缩） |
| MTransE 式联合训练（边际排序对齐损失 + 负采样） | 0.000（种子余弦 0.64 —— 不塌缩但对不齐） |
| 仅 log 出度/入度两维的直接余弦 | **0.1125** ← 结构信号本身是有的 |

即：**结构信息存在，但"把两个图分别嵌入再对齐"这条路在 DBP15K 上走不通**。
根因是 TransE 的目标 ‖h+r−t‖→0 不正交不变，两个独立收敛的空间之间不存在可用的
对齐映射；而联合训练下两个语言的关系词表**完全不相交**，对齐损失只约束 4,050 个
种子点，对 10 万实体的全局几何没有牵引力。

## 本模块的做法

不学嵌入，直接在**图**上传播对齐 —— 这是保结构匹配最直白的落地方式：

    设当前已匹配集合 M（初始 = 训练种子）。对左侧实体 i 定义"匹配邻居画像"
        P_L[i, k] = #{ i′ : (i, i′ 有边) 且 (i′, k) ∈ M }
    右侧同理得 P_R[j, k]。则
        S(i, j) = cos( P_L[i], P_R[j] )
    取 S 上的**互近邻高置信对**并入 M，进入下一轮。

为什么它有效：DBP15K 的两侧是同一份 DBpedia 的两个语言版本，**对齐实体在图上
同构** —— 若 i 与 j 对齐，则 i 的邻居匹配集合与 j 的邻居匹配集合自然重合。
因此只要种子覆盖率达到一定程度，一轮传播就能把对齐沿边扩散出去。

为什么只用**无关系标签**的画像：DBP15K 两侧的关系词表**完全不相交**
（en 侧 `http://dbpedia.org/ontology/...`，fr 侧 `http://fr.dbpedia.org/...`），
按关系名匹配是零信息。全部数据集共用同一条"关系无关"的代码路径，
也就保证了跨数据集的口径一致。

画像矩阵的规模：`P_L` 是 (n_L × |M|) 的稀疏矩阵，非零元数 ≈ Σ_{i′∈M} deg(i′)，
首轮约 4050×5 ≈ 2 万，末轮约 1.5 万×5 ≈ 7.5 万 —— 全程稀疏，内存无压力。
得分矩阵 n_L×n_R 永不物化，一律分块。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .data import KG, Pair

# 拟合缓存：同一次实验里 CUSSM / StructSim / StructPropK / LinFuse 会重复请求
# 完全相同的传播构造（最贵的一步）。缓存不改变任何数值，只避免重复付费。
_CACHE: dict = {}


def adjacency(kg: KG, directed: bool = False) -> sp.csr_matrix:
    """无向无权邻接（多重边去重）。返回 (n, n) float32 csr。"""
    T = np.asarray(kg.rel_triples, dtype=np.int64)
    if len(T) == 0:
        return sp.csr_matrix((kg.n, kg.n), dtype=np.float32)
    r = T[:, 0]
    c = T[:, 2]
    if not directed:
        r = np.concatenate([r, T[:, 2]])
        c = np.concatenate([c, T[:, 0]])
    A = sp.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(kg.n, kg.n))
    A.sum_duplicates()
    A.data[:] = 1.0
    return A.tocsr()


def _rownorm_sp(M: sp.csr_matrix) -> sp.csr_matrix:
    n = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    n[n == 0] = 1.0
    return sp.diags(1.0 / n) @ M


class StructProp:
    """种子锚定的迭代结构传播。

    参数
    ----
    rounds    : 传播轮数（0 表示只用种子做一轮画像，不迭代）
    conf_q    : 每轮入选"高置信匹配"的分位数阈值（在**互近邻**中取分位）
    max_new   : 每轮最多新增匹配对数（防止一轮引入大量错误并自我强化）
    chunk     : 分块大小
    """

    def __init__(self, rounds: int = 3, conf_q: float = 55.0, max_new: int = 3000,
                 chunk: int = 512, seed: int = 2026, exclude_seed_in_new: bool = True):
        self.rounds, self.conf_q, self.max_new = rounds, conf_q, max_new
        self.chunk, self.seed = chunk, seed
        self.exclude_seed_in_new = exclude_seed_in_new

    # ---------------- 拟合 ----------------
    def fit(self, pair: Pair) -> "StructProp":
        key = (pair.name, self.rounds, self.conf_q, self.max_new, self.chunk, self.seed)
        hit = _CACHE.get(key)
        if hit is not None:
            return hit
        self._fit(pair)
        if len(_CACHE) < 64:
            _CACHE[key] = self
        return self

    def _fit(self, pair: Pair) -> "StructProp":
        self.AL = adjacency(pair.left)
        self.AR = adjacency(pair.right)
        self.nL, self.nR = pair.left.n, pair.right.n
        self.history = []

        mL = pair.seeds[:, 0].astype(np.int64)
        mR = pair.seeds[:, 1].astype(np.int64)
        self.mL, self.mR = mL, mR          # 供 _pick 去重使用（必须在首轮之前赋值）
        self.n_seed = len(mL)
        self.PL, self.PR = self._profiles(mL, mR)
        self.nM = len(mL)

        # 逐轮：用当前画像打分 → 取互近邻高置信对 → 并入 M
        for r in range(self.rounds):
            S_stat = self._mutual_stats()
            add = self._pick(S_stat, r)
            if len(add) == 0:
                break
            mL = np.concatenate([mL, add[:, 0]])
            mR = np.concatenate([mR, add[:, 1]])
            self.history.append({"round": r + 1, "added": int(len(add)),
                                 "total": int(len(mL))})
            self.mL, self.mR = mL, mR
            self.PL, self.PR = self._profiles(mL, mR)
            self.nM = len(mL)
        self.mL, self.mR = mL, mR
        return self

    def _profiles(self, mL, mR):
        """匹配邻居画像：P[:, k] = 与第 k 个已匹配对左侧实体的边数。"""
        PL = self.AL[:, mL]
        PR = self.AR[:, mR]
        return _rownorm_sp(PL.tocsr()), _rownorm_sp(PR.tocsr())

    # ---------------- 打分 ----------------
    def score_block(self, left_idx: np.ndarray) -> np.ndarray:
        """(len(left_idx), n_R) 的余弦得分 —— 稀疏乘积后显式 .toarray()（块内物化）。"""
        Q = self.PL[left_idx]
        return (self.PR @ Q.T).T.toarray().astype(np.float32)

    # ---------------- 互近邻统计（分块 + 只算非零画像的行列）----------------
    def _mutual_stats(self):
        """返回 (hit, conf, mut)。

        **性能要点**：首轮时绝大多数实体的画像全为零（与任何种子都没有边），
        对这些行打分的意义为零、代价却照付。故先用 indptr 差找出**非零行/列**，
        只在 (非零左行 × 非零右列) 上算，再把下标映射回全局。实测把
        66,858×105,889 的整趟扫描缩到约 1.5 万×2.5 万，快一个数量级。
        """
        nlz = np.nonzero(np.diff(self.PL.indptr))[0]
        nrz = np.nonzero(np.diff(self.PR.indptr))[0]
        nL, nR = self.nL, self.nR
        hit = np.full(nL, -1, np.int64)
        conf = np.full(nL, -np.inf, np.float32)
        if len(nlz) == 0 or len(nrz) == 0:
            return hit, conf, np.zeros(nL, bool)
        PRs = self.PR[nrz]
        bestc = np.full(len(nrz), -np.inf, np.float32)
        argc = np.zeros(len(nrz), np.int64)
        for i in range(0, len(nlz), self.chunk):
            li = nlz[i:i + self.chunk]
            S = (PRs @ self.PL[li].T).T.toarray().astype(np.float32)
            a = np.argmax(S, axis=1)
            hit[li] = nrz[a]
            conf[li] = S[np.arange(len(li)), a]
            j = np.argmax(S, axis=0)
            v = S[j, np.arange(S.shape[1])]
            upd = v > bestc
            bestc[upd] = v[upd]
            argc[upd] = li[j[upd]]
            del S
        ok = hit >= 0
        mut = np.zeros(nL, bool)
        mut[ok] = (argc[np.searchsorted(nrz, hit[ok])] == np.nonzero(ok)[0])
        return hit, conf, mut

    def _pick(self, stat, r: int) -> np.ndarray:
        hit, conf, mut = stat
        cand = np.nonzero(mut)[0]
        if len(cand) == 0:
            return np.zeros((0, 2), np.int64)
        if self.conf_q is not None:
            thr = float(np.percentile(conf[cand], self.conf_q))
            cand = cand[conf[cand] > thr]
        if self.max_new and len(cand) > self.max_new:
            order = np.argsort(-conf[cand])[:self.max_new]
            cand = cand[order]
        if self.exclude_seed_in_new:
            done = set(zip(self.mL.tolist(), self.mR.tolist()))
            pairs = [(int(i), int(hit[i])) for i in cand if (int(i), int(hit[i])) not in done]
        else:
            pairs = [(int(i), int(hit[i])) for i in cand]
        return np.array(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), np.int64)
