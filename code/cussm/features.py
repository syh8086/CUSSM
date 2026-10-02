# -*- coding: utf-8 -*-
"""特征层 —— 所有方法共用的输入表示。

**为什么单独成层**：第 5.4.4 节要求"基线方法与本文方法的输入是一致的"。最彻底的
保证不是口头约定，而是让两者调用**同一个函数**产出同一份张量。因此本模块是唯一
的特征产地，任何方法都不得自行构造特征。

## 跨图可比性：本模块的第一原则

跨语言/跨模态对齐要求两侧特征**落在同一个可比空间**。两条硬性约束：

  (R1) **两侧原始维度必须相同**。否则对稀疏特征做随机投影时，同一个种子会生成
       两个**不同**的投影矩阵，余弦相似度被彻底打乱。实测：违反 R1 时，种子对齐上
       的直接余弦 Hits@1 只有 0.0017（随机水平）。
  (R2) **任何一个特征维都必须有跨图语义**。按"各侧频次前 k 的关系"建签名是错的
       —— 法语侧第 k 维与英语侧第 k 维毫无对应。改用**不依赖关系命名的结构量**
       （度、邻居度分布、无标签图上的扩散）与**按频次秩分箱的关系签名**。

## 两个视图（对应第 4 章的"两路由"）

    struct —— 结构视图（70 维）：跨图可比的图论量 + 邻域传播
    sem    —— 语义视图（2048 维）：属性字面量与表层名的字符 n-gram TF-IDF，
              以**固定的**散列宽度承载，保证两侧同维、同散列函数

设计约束：全部使用确定性散列（zlib.crc32）与固定种子，`PYTHONHASHSEED` 不影响结果。
"""
from __future__ import annotations

import zlib
from collections import Counter
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .data import KG

# 固定宽度：两侧必须一致（R1）
D_VAL = 1024          # 属性字面量字符 n-gram 散列宽度
D_NAME = 1024         # 实体表层名字符 n-gram 散列宽度
D_STRUCT = 70         # 结构视图维度（4 度 + 8 出邻居度箱 + 8 入邻居度箱 + …）
NB_DEG = 8            # 邻居度分箱数
NB_RANK = 5           # 关系频次秩分箱数
FIXED_DEG_SCALE = 8.0  # 度特征的固定缩放（log1p 后除以它，两侧同一常数）


# ---------------------------------------------------------------- 小工具
def _l2(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (M / n).astype(np.float32)


def _l2_sp(M: sp.csr_matrix) -> sp.csr_matrix:
    n = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    n[n == 0] = 1.0
    return sp.diags(1.0 / n) @ M


def _ngrams(text: str, ns=(3, 4)) -> Counter:
    t = " " + text.lower().strip() + " "
    c: Counter = Counter()
    for n in ns:
        for i in range(len(t) - n + 1):
            c[t[i:i + n]] += 1
    return c


def _hash_bow(texts, dim: int, ns=(3, 4), n_items: int | None = None) -> sp.csr_matrix:
    """字符 n-gram 哈希词袋（确定性，用 crc32 而非内建 hash）。

    **宽度 dim 必须由调用方给定固定值**，不得由数据决定 —— 这是约束 R1 的落点。
    """
    rows, cols, vals = [], [], []
    cache: dict[str, Counter] = {}
    for i, tx in enumerate(texts):
        if not tx:
            continue
        c = cache.get(tx)
        if c is None:
            c = _ngrams(tx, ns)
            if len(cache) < 500000:
                cache[tx] = c
        for g, v in c.items():
            rows.append(i)
            cols.append(zlib.crc32(g.encode("utf-8")) % dim)
            vals.append(float(v))
    M = sp.csr_matrix((vals, (rows, cols)),
                      shape=(n_items if n_items is not None else len(texts), dim),
                      dtype=np.float32)
    M.sum_duplicates()
    return M


def _tfidf(M: sp.csr_matrix) -> sp.csr_matrix:
    """按列（特征维）加权 TF-IDF：M @ diag(idf)。"""
    df = np.asarray((M > 0).sum(axis=0)).ravel().astype(np.float64)
    N = max(M.shape[0], 1)
    idf = np.log((1.0 + N) / (1.0 + df)) + 1.0
    return (M @ sp.diags(idf.astype(np.float32))).tocsr()


def rand_proj(d_in: int, d_out: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((d_in, d_out)) / np.sqrt(d_out)).astype(np.float32)


# ---------------------------------------------------------------- 结构视图
def structure_block(kg: KG) -> np.ndarray:
    """跨图可比的结构量（不依赖关系命名）。返回 (n, D_STRUCT) float32，行 L2 归一化。"""
    n = kg.n
    tri = kg.rel_triples
    h = tri[:, 0].astype(np.int64) if len(tri) else np.zeros(0, np.int64)
    r = tri[:, 1].astype(np.int64) if len(tri) else np.zeros(0, np.int64)
    t = tri[:, 2].astype(np.int64) if len(tri) else np.zeros(0, np.int64)

    do = np.bincount(h, minlength=n).astype(np.float32)
    di = np.bincount(t, minlength=n).astype(np.float32)
    deg = np.log1p(do + di)

    # ① 度特征（固定常数缩放，两侧同一尺度）
    b_deg = np.stack([np.log1p(do) / FIXED_DEG_SCALE,
                      np.log1p(di) / FIXED_DEG_SCALE,
                      deg / FIXED_DEG_SCALE,
                      (np.log1p(do) - np.log1p(di)) / FIXED_DEG_SCALE], axis=1)

    # ② 邻居度分布（log2 分箱，出/入各 NB_DEG 箱）
    bins = np.minimum((deg / 1.5).astype(np.int64), NB_DEG - 1) if len(tri) \
        else np.zeros(n, np.int64)
    hist_o = np.zeros((n, NB_DEG), np.float32)
    hist_i = np.zeros((n, NB_DEG), np.float32)
    if len(tri):
        np.add.at(hist_o, (h, bins[t]), 1.0)
        np.add.at(hist_i, (t, bins[h]), 1.0)
    b_no = _l2(hist_o)
    b_ni = _l2(hist_i)

    # ③ 关系频次**秩**分箱（不是关系名 —— 两侧的秩语义可比）
    b_rk = np.zeros((n, 2 * NB_RANK), np.float32)
    if len(tri):
        cnt = np.bincount(r)
        order = np.argsort(-cnt)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        rb = np.minimum(np.log2(rank + 1).astype(np.int64), NB_RANK - 1)
        np.add.at(b_rk, (h, rb[r]), 1.0)
        np.add.at(b_rk, (t, NB_RANK + rb[r]), 1.0)
    b_rk = _l2(b_rk)

    # ④ 邻域传播（无标签图上的两轮扩散 —— 跨图可比）
    h0 = np.hstack([b_deg, b_no, b_ni, b_rk]).astype(np.float32)
    if len(tri):
        A = sp.csr_matrix((np.ones(len(h) * 2, np.float32),
                           (np.concatenate([h, t]), np.concatenate([t, h]))), shape=(n, n))
        A.sum_duplicates()
        dn = np.asarray(A.sum(axis=1)).ravel()
        dn[dn == 0] = 1.0
        An = sp.diags(1.0 / dn) @ A
        h1 = np.asarray(An @ h0, dtype=np.float32)
        h2 = np.asarray(An @ h1, dtype=np.float32)
    else:
        h1 = np.zeros_like(h0)
        h2 = np.zeros_like(h0)
    Z = np.hstack([h0, _l2(h1), _l2(h2)]).astype(np.float32)
    # 补齐/截断到固定维度（不同 KG 的 h0 宽度恒为 4+8+8+10=30，故 3×30=90 恒定）
    if Z.shape[1] < D_STRUCT:
        Z = np.hstack([Z, np.zeros((n, D_STRUCT - Z.shape[1]), np.float32)])
    elif Z.shape[1] > D_STRUCT:
        Z = Z[:, :D_STRUCT]
    return _l2(Z)


# ---------------------------------------------------------------- 语义视图
def semantic_block(kg: KG) -> tuple[sp.csr_matrix, bool]:
    """属性字面量 + 表层名的字符 n-gram TF-IDF，固定宽度散列。

    返回**分开的两块**（属性值块、表层名块），各自 L2 归一化。**必须分开**：
    若先拼接再统一归一化，两块会互相稀释——实测 DBP15K fr_en 上"属性+名字"联合
    归一的 Hits@1 只有 0.635，而名字单独一路是 0.762，属性单独一路是 0.650。
    分开保留后，调用方（路由 K）可以按验证集选出合适的字面/属性配比。
    """
    n = kg.n
    vals = [[] for _ in range(n)]
    for e, _k, v, _lg in kg.att_triples:
        if 0 <= e < n:
            vals[e].append(v)
    val_text = [" ; ".join(vs) for vs in vals]
    surf = kg.surfaces()
    has_text = any(val_text) or any(surf)
    Xv = _l2_sp(_tfidf(_hash_bow(val_text, D_VAL, n_items=n)))
    Xn = _l2_sp(_tfidf(_hash_bow(surf, D_NAME, n_items=n)))
    return Xv.astype(np.float32), Xn.astype(np.float32), bool(has_text)


# ---------------------------------------------------------------- 主结构
@dataclass
class View:
    """一个 KG 的双视图特征。"""
    key: str
    struct: np.ndarray            # (n, d_s) float32，行 L2 归一化
    sem: np.ndarray               # (n, d_a) float32，行 L2 归一化
    deg_out: np.ndarray
    deg_in: np.ndarray
    has_text: bool
    dims: dict
    sem_val: np.ndarray | None = None    # 属性值块（已投影、已单独归一化）
    sem_name: np.ndarray | None = None   # 表层名块（已投影、已单独归一化）

    @property
    def n(self) -> int:
        return self.struct.shape[0]


def build_view(kg: KG, d_struct: int = 64, d_sem: int = 256,
               seed: int = 2026) -> View:
    """构建双视图。d_struct/d_sem 为**投影后**维度，投影矩阵两侧同种子同形状。

    **带构造缓存**：同一次实验里 NNSim / AttrSim / LinFuse / CUSSM / JAPE 都要同一份
    视图，而 fr_en 上一次 `build_view` 要 27 s（属性断言 30 万条 × 字符 3-4 gram）。
    缓存以"图身份 + 维度 + 种子"为键，命中则直接复用 —— 这**不改变任何数值**，
    只是不让同一份确定性计算重复付费。
    """
    key = (kg.name, kg.n, len(kg.rel_triples), d_struct, d_sem, seed,
           getattr(kg, "modality", None),
           # 语义视图覆盖值必须进缓存键：否则"先按几何建视图、后装编码器"的调用
           # 顺序会命中旧缓存，把编码器嵌入静默丢掉（本轮加，属正确性修正）。
           None if getattr(kg, "sem_override", None) is None
           else (int(np.shape(kg.sem_override)[0]), int(np.shape(kg.sem_override)[1])))
    hit = _VIEW_CACHE.get(key)
    if hit is not None:
        return hit
    v = _build_view_uncached(kg, d_struct, d_sem, seed)
    if len(_VIEW_CACHE) < 24:
        _VIEW_CACHE[key] = v
    return v


_VIEW_CACHE: dict = {}


def _build_view_uncached(kg: KG, d_struct: int, d_sem: int, seed: int) -> View:
    n = kg.n
    tri = kg.rel_triples

    # ---------------- 语义视图覆盖（跨模态编码器产物）----------------
    # Track A′ 装上 `cussm.mm_encoder` 的双塔后，两侧语义视图改由编码器给出；
    # 覆盖值必须是**已 L2 归一化**的 (n, d) 数组。若 d != d_sem，用固定随机投影
    # 升降维（两侧同种子同矩阵 —— 与约束 R1 是同一要求）。
    ov = getattr(kg, "sem_override", None)
    if ov is not None:
        Za = _l2(np.asarray(ov, dtype=np.float32))
        if Za.shape[0] != n:
            raise ValueError(f"sem_override 行数 {Za.shape[0]} != 实体数 {n}")
        if Za.shape[1] != d_sem:
            Za = _l2(Za @ rand_proj(Za.shape[1], d_sem, seed + 11))
        so = getattr(kg, "struct_override", None)
        if so is not None:
            Zs = _l2(np.asarray(so, dtype=np.float32))
        else:
            Sraw = structure_block(kg)
            Zs = _l2(Sraw @ rand_proj(Sraw.shape[1], d_struct, seed + 3))
        return View(kg.name, Zs, Za, np.zeros(n, np.float32),
                    np.zeros(n, np.float32), len(kg.att_triples) > 0,
                    {"struct": d_struct, "sem": d_sem,
                     "src": getattr(kg, "sem_override_note", None) or "encoder_override"},
                    sem_val=Za, sem_name=Za)

    # ---------------- 视觉侧（Flickr 区域）：几何构型 ----------------
    if getattr(kg, "modality", None) == "visual" and hasattr(kg, "geom"):
        g = np.asarray(kg.geom, dtype=np.float32)
        gs = np.hstack([g, g ** 2, np.sin(np.pi * g)])
        ga = np.hstack([g, np.sin(np.pi * g), np.cos(np.pi * g), g * g[:, ::-1]])
        Zs = _l2(gs @ rand_proj(gs.shape[1], d_struct, seed + 3))
        Za = _l2(ga @ rand_proj(ga.shape[1], d_sem, seed + 5))
        return View(kg.name, Zs, Za, np.zeros(n, np.float32),
                    np.zeros(n, np.float32), False,
                    {"struct": d_struct, "sem": d_sem, "src": "geom"},
                    sem_val=Za, sem_name=Za)

    # ---------------- 结构视图 ----------------
    Sraw = structure_block(kg)                    # (n, 70) 已可比
    Zs = _l2(Sraw @ rand_proj(Sraw.shape[1], d_struct, seed + 3))

    # ---------------- 语义视图：属性值块与表层名块**分别**投影、分别归一化 ----
    Xv, Xn, has_text = semantic_block(kg)         # 两块都是 (n, 1024) 稀疏、同散列
    Rv = rand_proj(Xv.shape[1], d_sem, seed + 5)  # 两侧必为同一矩阵（约束 R1）
    Rn = rand_proj(Xn.shape[1], d_sem, seed + 7)
    Zv = _l2(np.asarray(Xv @ Rv, dtype=np.float32))
    Zn = _l2(np.asarray(Xn @ Rn, dtype=np.float32))
    if not has_text:
        # 无文本时语义视图无信息：整块置零，由 Δ_nat 如实暴露"两路由不同"的事实
        Zv[:] = 0.0
        Zn[:] = 0.0
    Za = _l2(np.hstack([Zv, Zn]))

    deg_out = np.bincount(tri[:, 0], minlength=n).astype(np.float32) if len(tri) \
        else np.zeros(n, np.float32)
    deg_in = np.bincount(tri[:, 2], minlength=n).astype(np.float32) if len(tri) \
        else np.zeros(n, np.float32)
    return View(kg.name, Zs, Za, deg_out, deg_in, has_text,
                {"struct": d_struct, "sem": d_sem, "raw_struct": int(Sraw.shape[1]),
                 "raw_sem_val": int(Xv.shape[1]), "raw_sem_name": int(Xn.shape[1]),
                 "src": "structural+lexical"},
                sem_val=Zv, sem_name=Zn)


# ---------------------------------------------------------------- 打分原语
def cosine_block(ZL: np.ndarray, ZR: np.ndarray) -> np.ndarray:
    """两侧均已行归一化时的余弦得分矩阵 (m, N)。"""
    return (ZL @ ZR.T).astype(np.float32)


# ---------------------------------------------------------------- 打字投影 / 纤维
def kmeans(X: np.ndarray, k: int, seed: int = 2026, iters: int = 25) -> np.ndarray:
    """确定性 k-means（k-means++ 初始化 + Lloyd 迭代）。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = max(1, min(k, n))
    idx = [int(rng.integers(n))]
    d2 = ((X - X[idx[0]]) ** 2).sum(axis=1)
    for _ in range(1, k):
        p = d2 / max(d2.sum(), 1e-12)
        idx.append(int(rng.choice(n, p=p)))
    C = X[idx].copy()
    lab = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        dist = _cdist(X, C)                     # 分块计算，避免 n×k×d 中间张量
        new = np.argmin(dist, axis=1).astype(np.int32)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            sel = lab == j
            if sel.any():
                C[j] = X[sel].mean(axis=0)
    return lab


def _cdist(X: np.ndarray, C: np.ndarray, block: int = 8192) -> np.ndarray:
    """分块欧氏距离平方。"""
    out = np.empty((X.shape[0], C.shape[0]), dtype=np.float32)
    cn = (C ** 2).sum(axis=1)
    for i in range(0, X.shape[0], block):
        xb = X[i:i + block]
        out[i:i + block] = (xb ** 2).sum(axis=1)[:, None] - 2 * xb @ C.T + cn[None, :]
    np.maximum(out, 0, out=out)
    return out


def fibers_from(joined_left: np.ndarray, right: np.ndarray, k: int,
                seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    """两支数据合起来做打字投影，返回 (左侧纤维标签, 右侧纤维标签)。

    两侧共享同一套类型空间，纤维标签才可比（对应"统一基范畴"的设定）。
    """
    J = np.vstack([joined_left, right])
    lab = kmeans(J, k, seed=seed)
    return lab[:len(joined_left)], lab[len(joined_left):]
