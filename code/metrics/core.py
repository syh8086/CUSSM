# -*- coding: utf-8 -*-
"""指标的细化实现（第 5.3 节）。

统一口径：**所有排名都在完整候选集上计算**（DBP15K：目标 KG 全部实体，不做任何
候选截断或预筛），且分块进行以免物化 n_query × n_candidate 的完整矩阵。

    Hits@K      真值落在前 K 的查询占比（K = 1, 5, 10）
    MRR         平均倒数排名（真值排名的倒数之均值）
    MAP@10      单真值设定下 AP = 1/rank（rank ≤ 10 时），否则 0
    MedRank     真值排名的中位数
    纤维保持率   预测的对齐对落在同一纤维的占比（对应 4.3.3 的硬约束层）

**并列裁决必须与模型的真实输出一致**（否则 Hits@1 就不是"答对率"）。本模块取
`tie="consistent"`：并列时下标小者优先，与 `np.argmax` 的裁决**逐位相同**，于是
`Hits@1 = 1.0 ⟺ 模型实际发出的那一个预测就是真值`。另提供 `tie="optimistic"`
（并列一律记 rank 1，部分 EA 文献的口径）作为**上界列**一并报告。

为什么这不是吹毛求疵：实测（DBP15K fr_en，1500 查询，候选 105,889）
    结构路由 0 轮   argmax 命中 0.2627  |  乐观口径 0.5847
                   平均并列 **32,045** 个候选，74.1% 的行存在并列
    1 轮            argmax 0.2607  |  乐观 0.4633  |  平均并列 27,880
    2 轮            argmax 0.2640  |  乐观 0.4367  |  平均并列 27,456
结构画像是**整数计数**向量，孤立/稀疏实体的画像为零，真值分数为 0 时任何候选
都与之并列 —— 乐观口径会把这个"退化解"无条件记成 rank 1。对照：LinFuse 与 CUSSM
的融合得分无并列（平均并列 1.0，含并列的行 0.0%），两者两种口径完全相同。

**关于纤维保持率的定位**：按 4.3.3，纤维保持不进 SPS 的分量，而是独立列出的
**硬约束落实率**——它检验的是"实现是否真的把类型约束落到了解里"，而不是匹配质量。
"""
from __future__ import annotations

import ctypes
import sys
import time

import numpy as np

from cussm import dev


def ranks(matcher, pair, left_idx: np.ndarray, truths: np.ndarray,
          chunk: int = 512, mask_fiber: bool = False,
          tie: str = "consistent") -> np.ndarray:
    """分块计算真值排名（1 = 榜首）。

    `tie="consistent"`（默认）：并列按下标小者优先 —— 与 `np.argmax` 的裁决一致，
        故 `Hits@1 == 1` 当且仅当模型实际发出的预测命中真值。
    `tie="optimistic"`       ：并列一律记 rank 1（乐观上界，供与文献对齐）。
    """
    out = np.empty(len(left_idx), dtype=np.int64)
    lab = matcher.fiber_labels()
    lab_L, lab_R = lab if lab else (None, None)
    bk = dev.get()
    opt = (tie == "optimistic")
    for i in range(0, len(left_idx), chunk):
        li = left_idx[i:i + chunk]
        S = np.asarray(matcher.score_block(pair, li), dtype=np.float32)
        gt = truths[i:i + chunk]
        if mask_fiber and lab_L is not None:
            S = np.where(lab_L[li][:, None] != lab_R[None, :], np.float32(-1e30), S)
        # 排名统计下沉到后端（GPU 上这是热点）。
        #   optimistic：并列一律记 rank 1            → n_gt + 1
        #   consistent：并列且下标小者先被 argmax 取走 → n_gt + add + 1
        # 两者与改造前的 numpy 写法逐位一致（`add` 的定义见 dev.rank_and_ties）。
        n_gt, _, add, _ = bk.rank_and_ties(
            S, gt, need_tie=False, need_add=not opt, need_argmax=False)
        out[i:i + chunk] = (n_gt + 1) if opt else (n_gt + add + 1)
    return out


def evaluate(matcher, pair, chunk: int = 512, ks=(1, 5, 10),
             left_idx: np.ndarray | None = None) -> dict:
    """在完整候选集上评测一个已拟合方法。**单趟分块**算完全部指标。

    主指标取 `tie="consistent"`（模型实际输出口径）；并列率与乐观上界另列，
    便于审稿人核对"高 Hits@1 是不是并列白拿的"。
    """
    li = pair.test[:, 0] if left_idx is None else left_idx
    gt = pair.test[:, 1] if left_idx is None else None
    if gt is None:
        # 只评一个子集时（`left_idx`），逐条反查该左实体的真值。
        # 2026-09-28 修：原写作 `dict(...)[li]`，把数组/列表当作 dict 的键，
        # 必然抛 `TypeError: unhashable type: 'numpy.ndarray'` —— 该分支此前
        # 从未被正确执行过（主路径 `left_idx=None` 不受影响）。
        _m = {int(a): int(b) for a, b in pair.test}
        gt = np.asarray([_m[int(x)] for x in np.asarray(li).ravel()], dtype=np.int64)
    gt = np.asarray(gt)
    lab = matcher.fiber_labels()
    lab_L, lab_R = lab if lab else (None, None)

    n = len(li)
    r = np.empty(n, np.int64)
    r_opt = np.empty(n, np.int64)
    ties = np.empty(n, np.float64)
    keep = 0
    bk = dev.get()
    t0 = time.time()
    for i in range(0, n, chunk):
        blk = li[i:i + chunk]
        S = np.asarray(matcher.score_block(pair, blk), dtype=np.float32)
        g = gt[i:i + chunk]
        # 排名 / 并列 / argmax 一次算齐（后端下沉，含原逐行 add 的向量化版本）
        n_gt, n_tie, add, amax = bk.rank_and_ties(
            S, g, need_tie=True, need_add=True, need_argmax=lab_L is not None)
        r_opt[i:i + chunk] = n_gt + 1
        ties[i:i + chunk] = n_tie
        r[i:i + chunk] = n_gt + add + 1
        if lab_L is not None:
            keep += int((lab_L[blk] == lab_R[amax]).sum())
        del S
    dt = time.time() - t0

    res = {f"Hits@{k}": float((r <= k).mean()) for k in ks}
    res["MRR"] = float((1.0 / r).mean())
    res["MAP@10"] = float(np.where(r <= 10, 1.0 / r, 0.0).mean())
    res["MedRank"] = float(np.median(r))
    res["n_query"] = n
    res["score_seconds"] = round(dt, 2)
    res["ranks"] = r

    # —— 并列诊断（与主指标同时报告，不单独占篇幅）——
    res["Hits@1_乐观"] = float((r_opt <= 1).mean())
    res["MRR_乐观"] = float((1.0 / r_opt).mean())
    res["并列数均值"] = float(ties.mean())
    res["含并列行占比"] = float((ties > 1).mean())
    res["纤维保持率"] = (keep / n) if lab_L is not None else None
    return res



def retrieval_metrics(S: np.ndarray, truth: np.ndarray, ks=(1, 5, 10)) -> dict:
    """区域↔短语（Track A′）的检索口径：R@K 与 mAP。"""
    r = np.empty(len(truth), dtype=np.int64)
    for i in range(len(truth)):
        s_gt = S[i, truth[i]]
        r[i] = (S[i] > s_gt).sum() + 1
    out = {f"R@{k}": float((r <= k).mean()) for k in ks}
    out["mAP"] = float(np.where(r <= 10, 1.0 / r, 0.0).mean())
    out["MRR"] = float((1.0 / r).mean())
    out["n_query"] = int(len(truth))
    return out


# ---------------------------------------------------------------- 效率
def peak_rss_mb() -> float:
    """当前进程峰值常驻内存（MB）。Windows 走 GetProcessMemoryInfo，失败则退到
    tracemalloc 的 Python 层峰值，再不行返回 nan。"""
    try:
        if sys.platform.startswith("win"):
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            h = ctypes.windll.kernel32.GetCurrentProcess()
            fn = None
            for dll, name in ((ctypes.windll.kernel32, "K32GetProcessMemoryInfo"),
                              (getattr(ctypes.windll, "psapi", None),
                               "GetProcessMemoryInfo")):
                if dll is not None and hasattr(dll, name):
                    fn = getattr(dll, name)
                    break
            if fn is not None and fn(h, ctypes.byref(pmc), pmc.cb):
                if pmc.PeakWorkingSetSize > 0:
                    return pmc.PeakWorkingSetSize / 1048576.0
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:                                    # noqa: BLE001
        try:
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            return tracemalloc.get_traced_memory()[1] / 1048576.0
        except Exception:                                # noqa: BLE001
            return float("nan")
