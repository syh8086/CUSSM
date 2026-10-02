# -*- coding: utf-8 -*-
"""设备后端一致性测试（`cussm/dev.py` 的安全网）。

两类断言，性质完全不同：

**A. CPU 分支 == 改动前的 numpy 实现（逐位相等）。**
   参照物是**改造前的原始公式**（本文件里以 `_ref_*` 重写一遍），而不是"新代码
   自洽"。这一条保证后端抽象只改变了"在哪里算"，没有改变"算什么"——项目纪律
   要求主表与消融跑同一份代码路径，这一条就是它的机械保证。

**B. GPU 分支 ≈ CPU 分支（按容差，并检查排名与指标一致）。**
   跨设备的浮点归约顺序不同，逐位相等不可指望。判据取项目口径：**排名与指标
   一致**；同时报告最大绝对差与不一致元素占比，便于判断差异是否只是末位噪声。

用法：
    python code/test_device_parity.py                     # 自动（有 CUDA 则加测 GPU）
    CUSSM_DEVICE=cpu python code/test_device_parity.py     # 强制只测 CPU
    python code/test_device_parity.py --big               # 加测大矩阵（显存/内存压力）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cussm import dev                                        # noqa: E402
from metrics import core as mcore                           # noqa: E402

_PASS = 0
_FAIL = 0


def ck(name: str, cond: bool, note: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✓ {name}" + (f"  [{note}]" if note else ""))
    else:
        _FAIL += 1
        print(f"  ✗ {name}  {note}")


# ============================================================================
# 改造前的原始实现（参照物）
# ============================================================================
def _ref_rank_and_ties(S, gt):
    """`metrics/core.py` 改动前的 `evaluate` 内联逻辑，逐字抄回。"""
    S = np.asarray(S, dtype=np.float32)
    gt = np.asarray(gt).astype(np.int64).ravel()
    rows = np.arange(len(gt))
    s_gt = S[rows, gt]
    n_gt = (S > s_gt[:, None]).sum(axis=1)
    ties = (S == s_gt[:, None]).sum(axis=1)
    add = np.empty(len(gt), dtype=np.int64)
    for k in range(len(gt)):
        g = int(gt[k])
        add[k] = int(np.count_nonzero(S[k, :g] == s_gt[k]))
    amax = S.argmax(axis=1)
    return n_gt.astype(np.int64), ties.astype(np.int64), add, amax


def _ref_sinkhorn(C, eps=0.05, mask=None, n_iter=80, tol=1e-7):
    """`transport.sinkhorn` 改动前的原始实现。"""
    logK = -np.asarray(C, dtype=np.float64) / max(eps, 1e-9)
    if mask is not None:
        logK = np.where(mask, -1e12, logK)
    n1, n2 = logK.shape
    u = np.zeros(n1)
    v = np.zeros(n2)

    def lse_row(logK, v):
        M = logK + v[None, :]
        m = M.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(M - m).sum(axis=1, keepdims=True))).ravel()

    def lse_col(logK, u):
        M = logK + u[:, None]
        m = M.max(axis=0, keepdims=True)
        return (m + np.log(np.exp(M - m).sum(axis=0, keepdims=True))).ravel()

    for _ in range(n_iter):
        u_new = -lse_row(logK, v)
        v_new = -lse_col(logK, u_new)
        conv = (np.max(np.abs(u_new - u)) < tol and np.max(np.abs(v_new - v)) < tol)
        u, v = u_new, v_new
        if conv:
            break
    return np.exp(logK + u[:, None] + v[None, :]).astype(np.float32)


def _ref_mix_matmul(a1, b1, w1, a2, b2, w2):
    return (np.float32(w1) * (a1 @ b1).astype(np.float32)
            + np.float32(w2) * (a2 @ b2).astype(np.float32)).astype(np.float32)


# ============================================================================
# 数据构造
# ============================================================================
def make_int_scores(m: int, N: int, seed: int, ties: bool):
    """构造得分块。

    `ties=True` 用**小整数计数**（模拟结构画像）——这样行内必然出现大量并列，
    正是 `add`（并列且下标小于真值）这条分支唯一会生效的场合。不造这个场景，
    测试就只是"没覆盖到"。
    """
    rng = np.random.default_rng(seed)
    if ties:
        S = rng.integers(0, 7, size=(m, N)).astype(np.float32)
    else:
        S = rng.standard_normal((m, N)).astype(np.float32)
    gt = rng.integers(0, N, size=m).astype(np.int64)
    return S, gt


# ============================================================================
# 测试
# ============================================================================
def test_cpu_parity(args):
    print("\n=== A. CPU 分支 vs 改动前的原始实现（逐位）===")
    bk = dev.get("cpu")

    for tag, ties in (("无并列（连续得分）", False), ("大量并列（整数计数）", True)):
        S, gt = make_int_scores(300, 4000, seed=2026, ties=ties)
        n_gt_r, tie_r, add_r, amax_r = _ref_rank_and_ties(S, gt)
        n_gt, n_tie, add, amax = bk.rank_and_ties(
            S, gt, need_tie=True, need_add=True, need_argmax=True)
        ck(f"rank_and_ties.n_gt 逐位一致 [{tag}]", np.array_equal(n_gt, n_gt_r))
        ck(f"rank_and_ties.n_tie 逐位一致 [{tag}]", np.array_equal(n_tie, tie_r))
        ck(f"rank_and_ties.add 逐位一致 [{tag}]", np.array_equal(add, add_r))
        ck(f"rank_and_ties.argmax 逐位一致 [{tag}]", np.array_equal(amax, amax_r))
        # 排名口径的两条式子
        r_opt_r = n_gt_r + 1
        r_con_r = n_gt_r + add_r + 1
        ck(f"排名(乐观) 与原始一致 [{tag}]", np.array_equal(n_gt + 1, r_opt_r))
        ck(f"排名(argmax一致) 与原始一致 [{tag}]", np.array_equal(n_gt + add + 1, r_con_r))

    # sinkhorn（含掩码）
    rng = np.random.default_rng(7)
    C = rng.random((60, 80)).astype(np.float64)
    mask = rng.random((60, 80)) < 0.25
    for tag, mk in (("无掩码", None), ("带纤维掩码", mask)):
        P_ref = _ref_sinkhorn(C, eps=0.05, mask=mk)
        P = bk.sinkhorn(C, eps=0.05, mask=mk)
        ck(f"sinkhorn 逐位一致 [{tag}]", np.array_equal(P, P_ref),
           f"maxdiff={np.abs(P - P_ref).max():.3e}")
        # 长方形（60×80）：交替归一化以列方向收尾 ⇒ 只有**列和**为 1。
        # 旧断言写的是"行和为 1"，与实现不符（见 dev.sinkhorn 的实测记录）。
        ck(f"sinkhorn 列和为 1 [{tag}]",
           np.allclose(P.sum(axis=0), 1.0, atol=1e-5),
           f"max|colsum-1|={np.abs(P.sum(axis=0) - 1).max():.2e}")
    # 掩码位置必须被压到数值零（"约束在最优点内"的可观测后果）
    P_m = bk.sinkhorn(C, eps=0.05, mask=mask)
    ck("掩码位置的概率可忽略（< 1e-6）", P_m[mask].max() < 1e-6,
       f"max={P_m[mask].max():.2e}")

    # ---- 归一化的真实性质（2026-09-28 实测）----
    # 方阵：不动点存在，行和与列和**都**收敛到 1。
    Cq = rng.random((48, 48))
    Pq = bk.sinkhorn(Cq, eps=1.0, n_iter=4000)
    ck("sinkhorn 方阵：行和与列和均收敛到 1",
       np.allclose(Pq.sum(1), 1.0, atol=1e-5)
       and np.allclose(Pq.sum(0), 1.0, atol=1e-5),
       f"row={np.abs(Pq.sum(1) - 1).max():.2e} col={np.abs(Pq.sum(0) - 1).max():.2e}")
    # 长方形：不动点不存在（行和全 1 与列和全 1 不能并存），且加大迭代不改善。
    Cr = rng.random((60, 80))
    Pr = bk.sinkhorn(Cr, eps=0.05, n_iter=5000)
    ck("sinkhorn 长方形：只有列和为 1、行和不收敛（不动点不存在）",
       np.allclose(Pr.sum(0), 1.0, atol=1e-5)
       and not np.allclose(Pr.sum(1), 1.0, atol=1e-3),
       f"col={np.abs(Pr.sum(0) - 1).max():.2e} row={np.abs(Pr.sum(1) - 1).max():.2e}")
    ck("sinkhorn 长方形：行和偏差与 n_iter 无关（非收敛不足）",
       np.array_equal(Pr, bk.sinkhorn(Cr, eps=0.05, n_iter=80)))

    # mix_matmul
    a1 = rng.standard_normal((40, 64)).astype(np.float32)
    b1 = rng.standard_normal((64, 300)).astype(np.float32)
    a2 = rng.standard_normal((40, 64)).astype(np.float32)
    b2 = rng.standard_normal((64, 300)).astype(np.float32)
    for w in (0.0, 0.5, 1.0, 0.37):
        got = bk.mix_matmul(a1, b1, w, a2, b2, 1.0 - w)
        ref = _ref_mix_matmul(a1, b1, w, a2, b2, 1.0 - w)
        ck(f"mix_matmul 逐位一致 [w={w}]", np.array_equal(got, ref))

    # topk：与 argpartition 的语义是"集合相同"（顺序不影响调用点，那里会去重）
    S2 = rng.standard_normal((30, 500)).astype(np.float32)
    for k in (1, 5, 50):
        got = bk.topk_indices(S2, k)
        ref = np.argpartition(-S2, k - 1, axis=1)[:, :k]
        same = all(set(got[i].tolist()) == set(ref[i].tolist()) for i in range(len(S2)))
        ck(f"topk_indices 集合一致 [k={k}]", same)

    # ranks_of_truth
    S3, gt3 = make_int_scores(200, 3000, seed=99, ties=True)
    mask3 = np.zeros_like(S3, dtype=bool)
    got = bk.ranks_of_truth(S3, gt3, mask=mask3, chunk=64)
    ref = bk.rank_and_ties(S3, gt3, need_tie=False, need_add=False,
                           need_argmax=False)[0] + 1
    ck("ranks_of_truth 与 rank_and_ties 自洽", np.array_equal(got, ref))


def test_metrics_layer(args):
    """`metrics.core` 的公开函数在 CPU 上必须与原始公式给出同一组指标。"""
    print("\n=== A2. metrics.core 层（CPU）===")
    rng = np.random.default_rng(11)
    m, N = 250, 3000
    S_full = rng.integers(0, 6, size=(m, N)).astype(np.float32)   # 制造并列
    li = np.arange(m)
    gt = rng.integers(0, N, size=m).astype(np.int64)

    class MockMatcher:
        def score_block(self, pair, idx):
            return S_full[np.asarray(idx)]

        def fiber_labels(self):
            return None

    class MockPair:
        test = np.stack([li, gt], axis=1)

    got = mcore.evaluate(MockMatcher(), MockPair(), chunk=64)
    n_gt_r, tie_r, add_r, amax_r = _ref_rank_and_ties(S_full, gt)
    r_ref = n_gt_r + add_r + 1
    r_opt_ref = n_gt_r + 1
    ck("evaluate: Hits@1 与原始公式一致",
       abs(got["Hits@1"] - float((r_ref <= 1).mean())) < 1e-12,
       f"{got['Hits@1']:.6f}")
    ck("evaluate: MRR 与原始公式一致",
       abs(got["MRR"] - float((1.0 / r_ref).mean())) < 1e-12)
    ck("evaluate: Hits@1_乐观 与原始公式一致",
       abs(got["Hits@1_乐观"] - float((r_opt_ref <= 1).mean())) < 1e-12)
    ck("evaluate: 并列数均值 与原始公式一致",
       abs(got["并列数均值"] - float(tie_r.mean())) < 1e-9)
    ck("evaluate: 排名数组逐位一致", np.array_equal(got["ranks"], r_ref))

    # ranks() 的两种口径
    r_cons = mcore.ranks(MockMatcher(), MockPair(), li, gt, chunk=64,
                         tie="consistent")
    r_opt = mcore.ranks(MockMatcher(), MockPair(), li, gt, chunk=64,
                        tie="optimistic")
    ck("ranks(tie=consistent) 逐位一致", np.array_equal(r_cons, r_ref))
    ck("ranks(tie=optimistic) 逐位一致", np.array_equal(r_opt, r_opt_ref))


def test_gpu_parity(args):
    print("\n=== B. GPU 分支 ≈ CPU 分支（容差 + 排名一致）===")
    if not dev.cuda_available():
        print("  (无可用 CUDA —— 跳过；这不影响 A 类断言的效力)")
        return
    bk_gpu = dev.get("cuda")
    bk_cpu = dev.get("cpu")

    # 用**整数计数**矩阵：并列多，且比较是全等的，GPU/CPU 应完全一致
    S, gt = make_int_scores(400, 20000, seed=2026, ties=True)
    ng_c, nt_c, ad_c, am_c = bk_cpu.rank_and_ties(S, gt, True, True, True)
    ng_g, nt_g, ad_g, am_g = bk_gpu.rank_and_ties(S, gt, True, True, True)
    ck("GPU: n_gt 逐位一致", np.array_equal(ng_g, ng_c))
    ck("GPU: n_tie 逐位一致", np.array_equal(nt_g, nt_c))
    ck("GPU: add 逐位一致", np.array_equal(ad_g, ad_c))
    ck("GPU: argmax 逐位一致（并列裁决同 idx 小者）", np.array_equal(am_g, am_c))

    # 连续得分：跨设备只要求排名口径一致
    S2, gt2 = make_int_scores(400, 20000, seed=5, ties=False)
    ng2_c, _, ad2_c, _ = bk_cpu.rank_and_ties(S2, gt2, False, True, False)
    ng2_g, _, ad2_g, _ = bk_gpu.rank_and_ties(S2, gt2, False, True, False)
    r_c, r_g = ng2_c + ad2_c + 1, ng2_g + ad2_g + 1
    agree = float((r_c == r_g).mean())
    ck("GPU: 真值排名一致率 ≥ 0.999", agree >= 0.999,
       f"{agree:.6f}（最大位次差 {np.abs(r_c - r_g).max()}）")
    ck("GPU: Hits@1 一致",
       abs(float((r_c <= 1).mean()) - float((r_g <= 1).mean())) < 1e-12)

    # sinkhorn
    rng = np.random.default_rng(3)
    C = rng.random((200, 300)).astype(np.float64)
    mask = rng.random((200, 300)) < 0.2
    P_c = bk_cpu.sinkhorn(C, 0.05, mask)
    P_g = bk_gpu.sinkhorn(C, 0.05, mask)
    ck("GPU: sinkhorn 最大绝对差 < 1e-6",
       float(np.abs(P_c - P_g).max()) < 1e-6,
       f"{np.abs(P_c - P_g).max():.3e}")
    ck("GPU: sinkhorn 的 argmax 逐位一致",
       np.array_equal(P_c.argmax(1), P_g.argmax(1)))

    # mix_matmul（跨设备：归约顺序不同 ⇒ 允许按 float32 舍入量级偏离）
    a1 = rng.standard_normal((300, 512)).astype(np.float32)
    b1 = rng.standard_normal((512, 20000)).astype(np.float32)
    a2 = rng.standard_normal((300, 512)).astype(np.float32)
    b2 = rng.standard_normal((512, 20000)).astype(np.float32)
    M_c = bk_cpu.mix_matmul(a1, b1, 0.4, a2, b2, 0.6)
    M_g = bk_gpu.mix_matmul(a1, b1, 0.4, a2, b2, 0.6)
    #
    # 阈值必须**由浮点性质推出**，不能拍一个数。推导：
    #   u = 2^-24 ≈ 5.96e-8（float32 单位舍入）；k = 归约长度（此处 512）。
    #   CPU 与 GPU 对同一行做**不同次序**的求和（SIMD 分块 vs split-K/tile），
    #   各步舍入按统计口径约累积 u·√k；取 10× 安全系数 ⇒ 阈值 10·u·√k ≈ 1.35e-5。
    #   （经典上界 γ_k = k·u ≈ 3.05e-5 假设误差同号全累积，过于悲观；用它等于放弃检验。
    #     2026-09-28 实测：本项 maxabs=8.39e-5、约为阈值的 8%，属正常量级。）
    #   分母取 Σ_i|a_i·b_i| 的最大值——这是**前向误差的正确尺度**。此前用「结果最大值」
    #   作分母，在存在抵消时会低估条件数，把正常偏差误判为缺陷（本轮即因此报过一次假警）。
    k = a1.shape[1]
    u32 = 2.0 ** -24
    thr = 10.0 * u32 * float(np.sqrt(k))
    scale = max(float((np.abs(a1) @ np.abs(b1)).max()),
                float((np.abs(a2) @ np.abs(b2)).max()))
    mae = float(np.abs(M_c - M_g).max())
    rel = mae / max(scale, 1e-9)
    ck("GPU: mix_matmul 偏差在 float32 舍入量级内", rel < thr,
       f"maxabs={mae:.3e} /term_scale={rel:.2e} < {thr:.2e}")

    # 精度口径必须已被显式钉死：cudnn 的 TF32 会把 ViT 的 Conv2d patch embedding
    # 压到 10 位尾数，使"GPU 数值"混入两种精度语义，且默认值随 torch 版本漂移。
    from cussm import dev as _dev
    _p = _dev.lock_precision()
    ck("精度口径已钉死：matmul 与 cudnn 的 TF32 均关闭",
       _p.get("matmul.allow_tf32") is False
       and _p.get("cudnn.allow_tf32") is False,
       json.dumps(_p, ensure_ascii=False))


def test_big(args):
    if not args.big:
        return
    print("\n=== C. 大矩阵压力（fr_en 真实规模的候选池）===")
    N = 105889
    m = 512
    print(f"  候选池 N={N}，块宽 m={m}，单块 float32 = {m * N * 4 / 2**20:.0f} MB")
    bk = dev.get()
    S = (np.random.default_rng(1).standard_normal((m, N)) * 0.1).astype(np.float32)
    gt = np.random.default_rng(2).integers(0, N, size=m).astype(np.int64)
    import time
    t0 = time.time()
    n_gt, n_tie, add, amax = bk.rank_and_ties(S, gt, True, True, True)
    dt = time.time() - t0
    print(f"  rank_and_ties（{bk.name}）：{dt:.3f} s   "
          f"n_gt[0]={n_gt[0]} n_tie[0]={n_tie[0]} add[0]={add[0]}")
    ck("大矩阵下 add ≤ n_tie 且 n_gt + n_tie ≤ N",
       bool((add <= n_tie).all() and (n_gt + n_tie <= N).all()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="加测 105,889 候选池规模")
    ap.add_argument("--device", default=None, help="强制设备：cpu / cuda / auto")
    args = ap.parse_args()

    if args.device:
        os.environ[dev.ENV_KEY] = args.device
    dev.reset_cache()

    print("=" * 74)
    print(f"设备后端一致性测试   解析设备 = {dev.resolve_name(args.device)}"
          f"   torch={'有' if dev.torch_available() else '无'}"
          f"   cuda={'可用' if dev.cuda_available() else '不可用'}")
    print("=" * 74)

    test_cpu_parity(args)
    test_metrics_layer(args)
    test_gpu_parity(args)
    test_big(args)

    print("\n" + "=" * 74)
    print(f"结果：PASS {_PASS} / FAIL {_FAIL}")
    print("=" * 74)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
