# -*- coding: utf-8 -*-
"""最优传输与指派求解。

CUSSM 的匹配层在此：纤维约束通过**掩码**进入传输问题，而不是事后过滤 ——
这正是第 4.2 节"纤维保持的 lax 函子"在算法上的落地：不允许的跨纤维配对
在传输前就被置为不可达，因此得到的是"约束在最优点内"的解，而非"解完再纠正"。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import dev


def logsumexp(M: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(M, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.sum(np.exp(M - m), axis=axis, keepdims=True))).squeeze(axis)


def sinkhorn(C: np.ndarray, eps: float = 0.05, mask: np.ndarray | None = None,
             n_iter: int = 80, tol: float = 1e-7) -> np.ndarray:
    """熵正则最优传输，log 域稳定实现。

    C    : (n1,n2) 代价矩阵（越小越好）
    mask : (n1,n2) bool，True 表示**禁止**该配对（纤维约束）
    返回 : (n1,n2) 软指派（float32）。**归一化性质见下，勿按"行归一化"理解**。

    实现下沉到 `cussm.dev` 的后端（CPU→numpy，GPU→torch），**迭代式完全相同**：
    同为 log 域、u/v 交替更新、按 tol 提前停，且在 **float64** 上迭代（出口转
    float32），故两种设备上的收敛路径一致，差异仅来自归约顺序的末位。

    掩码语义不变：**禁止的配对被置为 −1e12 后进入迭代**，即"约束在最优点内"，
    而不是解完再过滤——这是 4.2 的纤维保持落到算法上的关键一步。

    ## 归一化：方阵才行，长方形必然只归一化一侧

    本函数做的是**无边缘约束**的行列交替归一化，且以列方向收尾：

    * n1 == n2（方阵）：不动点存在，行和与列和**都**收敛到 1；
    * n1 ≠ n2（长方形）：**不动点不存在**（"行和全 1"与"列和全 1"不能并存），
      输出只有**列和 = 1**，行和是残余值，且**加大迭代次数不改善**。

    本项目的调用点 `CUSSM._route_C` 传的是长方形矩阵，但只用其**行内相对序**
    （`argsort`），缩放不影响排序，故无害。详见 `cussm/dev.py` 的实测记录。
    """
    return dev.get().sinkhorn(C, eps=eps, mask=mask, n_iter=n_iter, tol=tol)


def _lse_row(logK: np.ndarray, v: np.ndarray) -> np.ndarray:
    """log-sum-exp（按行）。保留为公开别名：CPU 分支的内核在 `cussm.dev`。"""
    return dev._lse_row_np(logK, v)


def _lse_col(logK: np.ndarray, u: np.ndarray) -> np.ndarray:
    """log-sum-exp（按列）。保留为公开别名：CPU 分支的内核在 `cussm.dev`。"""
    return dev._lse_col_np(logK, u)


def greedy_match(S: np.ndarray, mask: np.ndarray | None = None,
                 chunk: int = 256) -> np.ndarray:
    """按得分贪心取每行最大（可分块，用于大候选集）。返回 (n1,) 目标下标。"""
    n1, n2 = S.shape
    out = np.empty(n1, dtype=np.int64)
    for i in range(0, n1, chunk):
        blk = S[i:i + chunk]
        if mask is not None:
            blk = np.where(mask[i:i + chunk], -np.inf, blk)
        out[i:i + chunk] = np.argmax(blk, axis=1)
    return out


def hungarian_match(S: np.ndarray) -> np.ndarray:
    """匈牙利指派（仅在 |左侧| ≤ |右侧| 且规模可控时使用）。"""
    r, c = linear_sum_assignment(-np.asarray(S, dtype=np.float64))
    out = np.full(S.shape[0], -1, dtype=np.int64)
    out[r] = c
    return out


def ranks_of_truth(S: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None,
                   chunk: int = 512) -> np.ndarray:
    """逐查询计算真值在候选集中的排名（1 = 榜首）。

    分块进行，不物化完整排好序的矩阵；`truth[k]` 为第 k 个查询的真值候选下标。
    实现下沉到 `cussm.dev` 的后端（GPU 上该运算是热点之一）。
    """
    return dev.get().ranks_of_truth(S, truth, mask=mask, chunk=chunk)
