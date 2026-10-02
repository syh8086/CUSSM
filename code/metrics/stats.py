# -*- coding: utf-8 -*-
"""统计口径（第 5.3.6 节）。

三项都用同一份**逐查询**的 0/1 命中向量，因此是真正的配对比较：

    bootstrap_ci          对逐查询指标重采样 B 次，给出 95% 百分位区间
    paired_permutation    配对置换检验：在"两法无差异"的原假设下随机交换每一对
                          的标签，得到差异统计量的零分布，再取双侧 p 值
    holm                  Holm–Bonferroni 逐步下降，对"本文方法 vs 每个基线"的多重
                          比较做族错误率控制

**为什么用置换而不是 t 检验**：Hits@1 是伯努利变量的均值，其配对差分布严重偏斜
且取值离散，t 检验的正态假设不成立；置换检验只依赖可交换性，是对该设定更保守的
选择。
"""
from __future__ import annotations

import numpy as np


def hit_vector(r: np.ndarray, k: int = 1) -> np.ndarray:
    return (np.asarray(r) <= k).astype(np.float64)


def bootstrap_ci(v: np.ndarray, B: int = 2000, alpha: float = 0.05,
                 seed: int = 2026) -> tuple:
    """对逐查询 0/1 向量做 bootstrap。

    返回 **三元组 `(下界, 上界, 点估计)`** —— 注意点估计在**第三位**，
    不是第一位（调用方请写 `lo, hi, point = bootstrap_ci(...)`，勿写成
    `point, lo, hi`，否则区间会看起来上下颠倒）。
    """
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi), float(v.mean()))


def paired_permutation(a: np.ndarray, b: np.ndarray, B: int = 10000,
                       seed: int = 2026) -> dict:
    """配对置换检验：H0 为两法在每一查询上的期望相同。返回差异与双侧 p 值。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return {"diff": 0.0, "p": 1.0, "n": 0}
    if np.allclose(d, 0):
        return {"diff": 0.0, "p": 1.0, "n": n}
    signs = rng.integers(0, 2, size=(B, n)) * 2.0 - 1.0
    null = (signs * d[None, :]).mean(axis=1)
    p = float((np.abs(null) >= abs(obs) - 1e-15).mean())
    return {"diff": obs, "p": max(p, 1.0 / B), "n": n}


def holm(pvals: list[float], names: list[str], alpha: float = 0.05) -> list[dict]:
    """Holm–Bonferroni 逐步校正。返回按原始 p 升序排列的判定表。"""
    m = len(pvals)
    order = np.argsort(pvals)
    out, prev_reject = [], True
    for rank, i in enumerate(order):
        thr = alpha / (m - rank)
        rej = bool(pvals[i] <= thr) and prev_reject
        if not rej:
            prev_reject = False
        out.append({"对照": names[i], "p": float(pvals[i]), "阈值": float(thr),
                    "显著": rej})
    return out
