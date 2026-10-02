#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 5 章跨数据集实测数据的**唯一出处**（结构覆盖率与路由判别力）。

## 为什么单独有这个脚本

第 5 章的表 12–17 里有若干数字（结构覆盖率、各路由单独判别力、"纯语义基线"、
同名/异名子集分解）不是 `run_experiment.py` 的主表产物。这些数字先前散落在若干
一次性诊断脚本的**屏幕输出**里，没有落盘、口径也不统一（"纯语义基线"一处按
"单通道最好"算、一处按"α=0 的融合"算），无法作为论文数值的来源。

本脚本把它们收进**一个**脚本、**一套**口径、**一份**落盘产物，且全部可重跑：

## 口径（唯一，不再有第二个版本）

所有判别力一律取 **`argmax` 口径**（＝模型实际输出，并列按下标小者裁决），
与 `metrics/core.py` 的 `tie="consistent"` 逐位一致。

- **结构覆盖率** = 结构画像非零的左侧实体占比。画像非零 ⟺ 该实体与至少一个训练
  种子有边。覆盖不到的查询，结构路由必然失分，故这是它的**绝对上界**。
- **H 单独** = 结构路由原始得分 `argmax` 的 Hits@1（单路得分做逐行 z-score 不改变
  `argmax`，故不必标准化）。
- **AN / AV** = 语义路由两个子通道（表层名块 / 属性值块）**原始读数**各自的 Hits@1。
- **K 单独（＝纯语义基线）** = `w*AN + (1−w)*AV`，`w` 在**验证种子**上选，
  报测试集 Hits@1。**这才是"不用结构路由"的正确基线**——用 `max(AN, AV)` 当基线
  是错的（两块可以互补，融合值通常高于任一单块）。
- **融合** = `α*z(H) + (1−α)*z(K)`，`(w, α)` 均在**验证种子**上选，报测试集 Hits@1。
  `α` 网格取 `cussm.model.ALPHA_GRID`（与 LinFuse 共用同一张，见该常量注释）。
- **结构路由净贡献** = 融合 − K 单独。

**关键纪律**：`(w, α)` 只在验证集上选，测试集只用来读数。曾经的一版诊断脚本在
测试集上直接搜 `(w, α)` 的最优格子，那等于把测试集当训练集用，本脚本不再这样做。

## 用法

    python code/measure_ch5.py                          # 全部 7 个可执行数据集（不含 Track A′）
    python code/measure_ch5.py --datasets wn18 fb15k237
    python code/measure_ch5.py --max-test 1000 --rounds 0

产物：`results/ch5_measure.json`、`results/CH5_MEASURE.md`

**不做的事**：本脚本不训练任何 torch 模型。TransE-NN / MTransE / JAPE /
迭代结构传播等需要训练的数值由 `run_experiment.py` 给出，两者共同构成第 5 章
的全部实测列。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from baselines.models import AttrSim, LinFuse, NNSim, StructSim        # noqa: E402
from metrics import core as mcore                                      # noqa: E402
from cussm import data as sdata                                         # noqa: E402
from cussm import mem                                                   # noqa: E402
from cussm.model import ALPHA_GRID, CUSSM, _zscore                       # noqa: E402
from cussm.propagation import StructProp                                # noqa: E402

# **可执行**的数据集：本脚本只在这些数据集上取数。
#
# `flickr30k_entities`（Track A′）**刻意不在其中**，但理由与"不可评估"无关——
# 该轨道已在 2026-09-28 按 5.1.1 逐条改造完毕（补空间关系边、双塔共享嵌入空间、
# 三重防泄漏）并**已实跑**，读数为表 18，由 `run_experiment.py --mm-encoder`
# 产出 `results/track_a_prime.json`。它不进本脚本的原因只有一条：**度量通路不匹配**——
# ① 该轨道的候选池为 1.9×10^5 条短语（对比 KG 轨道的 10^3~10^5 实体），本脚本的
#    稠密 `(n_q, N)` 打分与逐块 `argsort` 叠加在这个量级上会把内存打到 20 GB；
# ② 本脚本的三类度量（结构覆盖率、路由判别力、表层名分解）是为**跨语言 KG 对齐**
#    设计的，对区域—短语轨道无对应语义。
# 故该轨道不进入本脚本、也不进入本脚本的产物 `ch5_measure.json`；它的数值见表 18。
ALL_DATASETS = ["dbp15k_fr_en", "dbp15k_zh_en", "dbp15k_ja_en",
                "countries_s1", "fb15k237", "wn18", "yago3_10"]

# 各加载器支持的调试参数（显式列出，不用 try/except TypeError 兜底 —— 那会把
# 真正的参数错误一起吞掉）
_ACCEPTS_MAX_TEST = {"dbp15k_fr_en", "dbp15k_zh_en", "dbp15k_ja_en",
                     "flickr30k_entities", "fb15k237", "wn18", "yago3_10"}
_ACCEPTS_MAX_IMG = {"flickr30k_entities"}

W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


# ---------------------------------------------------------------- 工具
def _j(o):
    """把结果里的 numpy 标量/数组变成可 JSON 序列化的类型。"""
    if isinstance(o, dict):
        return {k: _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_j(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _j(o.tolist())
    return o


def _load(key: str, max_test: int | None, max_images: int | None, seed: int):
    kw = {"seed": seed}
    if max_test and key in _ACCEPTS_MAX_TEST:
        kw["max_test"] = max_test
    if max_images and key in _ACCEPTS_MAX_IMG:
        kw["max_images"] = max_images
    return sdata.LOADERS[key](**kw)


def _hits1_from_ranks(r: np.ndarray, mask: np.ndarray | None = None) -> float:
    v = (np.asarray(r) <= 1)
    return float(v.mean() if mask is None else v[mask].mean())


# ---------------------------------------------------------------- 单数据集测量
def measure(key: str, args) -> dict:
    t0 = time.time()
    print(f"\n{'=' * 78}\n### {key}\n{'=' * 78}", flush=True)
    pair = _load(key, args.max_test, args.max_images, args.seed)
    print("  " + pair.summary(), flush=True)

    li = pair.test[:, 0]
    gt = pair.test[:, 1]
    n_q = len(li)

    rec = {
        "dataset": key,
        "pair": pair.summary(),
        "meta": {k: v for k, v in pair.meta.items() if k not in ("att_left", "att_right")},
        "scale": {
            "n_ent_L": int(pair.left.n), "n_ent_R": int(pair.right.n),
            "n_rel_L": int(pair.left.n_rel), "n_rel_R": int(pair.right.n_rel),
            "n_rel_triples_L": int(len(pair.left.rel_triples)),
            "n_rel_triples_R": int(len(pair.right.rel_triples)),
            "n_att_L": int(len(pair.left.att_triples)),
            "n_att_R": int(len(pair.right.att_triples)),
            "n_seeds": int(len(pair.seeds)), "n_val": int(len(pair.val)),
            "n_test": int(len(pair.test)),
            "candidates": "目标侧全部实体（不截断）",
        },
    }

    # ---------------- ① 结构覆盖率 ----------------
    sp = StructProp(rounds=0, chunk=args.chunk, seed=args.seed).fit(pair)
    nz = np.diff(sp.PL.indptr) > 0
    rec["coverage"] = {
        "画像非零的左侧实体占比": float(nz.mean()),
        "画像非零的左侧实体数": int(nz.sum()),
        "测试查询画像全零占比": float(1.0 - nz[li].mean()),
        "PL_nnz": int(sp.PL.nnz),
        "PR_nnz": int(sp.PR.nnz),
        "种子对数": int(sp.n_seed),
        "传播轮数": 0,
    }
    print(f"  ① 结构覆盖率 {nz.mean():.4f}；测试查询画像全零 "
          f"{1 - nz[li].mean():.4f}", flush=True)

    # ---------------- ② 表层名分解（同名字集 vs 异名子集）----------------
    sl = pair.left.surfaces()
    sr = pair.right.surfaces()
    surf_L = np.array([sl[int(i)] for i in li], dtype=object)
    surf_R = np.array([sr[int(j)] for j in gt], dtype=object)
    same = np.array([str(a) == str(b) for a, b in zip(surf_L, surf_R)], dtype=bool)
    nn = NNSim(seed=args.seed).fit(pair)
    r_nn = mcore.ranks(nn, pair, li, gt, chunk=args.chunk, tie="consistent")
    rec["surface"] = {
        "测试对中两侧表层名完全相同的占比": float(same.mean()),
        "同名子集_NNSim_Hits@1": _hits1_from_ranks(r_nn, same) if same.any() else None,
        "异名子集_NNSim_Hits@1": _hits1_from_ranks(r_nn, ~same) if (~same).any() else None,
        "同名子集查询数": int(same.sum()),
        "异名子集查询数": int((~same).sum()),
    }
    print(f"  ② 同名占比 {same.mean():.4f}；NNSim 同名 "
          f"{rec['surface']['同名子集_NNSim_Hits@1']} / 异名 "
          f"{rec['surface']['异名子集_NNSim_Hits@1']}", flush=True)

    # ---------------- ③ 路由分解（(w, α) 只在验证集上选）----------------
    # 这里只需要语义路由的两个子通道读数，故把结构路由、纤维、类型、粘合全关掉 ——
    # 既避免触发昂贵的传播与聚类（本脚本要在 8 个数据集上跑），也不影响读数：
    # 结构路由的单独表现由第 ① 步的 StructProp(rounds=0) 直接给出。
    m = CUSSM(seed=args.seed, tune=False, use_route_H=False, use_fiber=False,
             use_type=False, use_glue=False).fit(pair)
    vi, vj = pair.val[:, 0], pair.val[:, 1]
    # **按行分块**（2026-09-29 消除 OOM）：原实现对全量 val / 全量 test 直接物化
    # (n_q, N) 的 H/AN/AV/K/zH/zK/fused —— yago3_10（n_val=3218、N=109,398）算术和
    # 达 11 GB，Track A′ 的候选池量级更大。改为按行块推进、**整数**累加命中数，
    # 峰值与查询数解耦。z-score 与 argmax 逐行独立，故选出的 (w, α) 与读数逐位不变。
    n_v = int(len(vi))
    CH = mem.plan_block(m.VR.n, arrays=6, chunk=args.chunk)

    acc_w = [0] * len(W_GRID)
    for sl in mem.block_slices(n_v, CH):
        Av, Bv = m._k_parts(vi[sl])
        vb = vj[sl]
        for i, w in enumerate(W_GRID):
            acc_w[i] += int((np.argmax(w * Av + (1.0 - w) * Bv, axis=1) == vb).sum())
        del Av, Bv
    bw, bh = 0.5, -1.0
    for i, w in enumerate(W_GRID):
        h = acc_w[i] / max(n_v, 1)
        if h > bh + 1e-12:
            bw, bh = float(w), h

    acc_a = [0] * len(ALPHA_GRID)
    acc0 = 0
    for sl in mem.block_slices(n_v, CH):
        zb = vi[sl]
        zHv = _zscore(sp.score_block(zb))
        Av, Bv = m._k_parts(zb)
        zKv = _zscore(bw * Av + (1.0 - bw) * Bv)
        vb = vj[sl]
        acc0 += int((np.argmax(zKv, axis=1) == vb).sum())
        for i, a in enumerate(ALPHA_GRID):
            acc_a[i] += int((np.argmax(a * zHv + (1.0 - a) * zKv, axis=1) == vb).sum())
        del zHv, zKv, Av, Bv
    ba, bh2 = 0.0, acc0 / max(n_v, 1)
    for i, a in enumerate(ALPHA_GRID):
        h = acc_a[i] / max(n_v, 1)
        if h > bh2 + 1e-12:
            ba, bh2 = float(a), h

    n_t = int(len(li))
    CHt = mem.plan_block(m.VR.n, arrays=8, chunk=args.chunk)
    nH = nAN = nAV = nK = nF = nDegen = n_tie = 0
    for sl in mem.block_slices(n_t, CHt):
        zb, gb = li[sl], gt[sl]
        H = sp.score_block(zb)
        AN, AV = m._k_parts(zb)
        K = bw * AN + (1.0 - bw) * AV
        zH, zK = _zscore(H), _zscore(K)
        fused = ba * zH + (1.0 - ba) * zK
        nH += int((np.argmax(H, axis=1) == gb).sum())
        nAN += int((np.argmax(AN, axis=1) == gb).sum())
        nAV += int((np.argmax(AV, axis=1) == gb).sum())
        nK += int((np.argmax(K, axis=1) == gb).sum())
        nF += int((np.argmax(fused, axis=1) == gb).sum())
        nDegen += int(((zK.max(axis=1) - zK.min(axis=1)) <= 0.0).sum())
        n_tie += int((fused == fused[np.arange(len(gb)), gb][:, None]).sum())
        del H, AN, AV, K, zH, zK, fused
    h_H = nH / max(n_t, 1)
    h_AN = nAN / max(n_t, 1)
    h_AV = nAV / max(n_t, 1)
    h_K = nK / max(n_t, 1)
    h_F = nF / max(n_t, 1)
    degen_mean = nDegen / max(n_t, 1)
    tie_mean = n_tie / max(n_t, 1)
    r_F = mcore.ranks(_FusedView(m, sp, bw, ba), pair, li, gt, chunk=args.chunk)
    rec["routes"] = {
        "w 字面权重（验证集选出）": bw, "w 的验证集 Hits@1": bh,
        "α 结构权重（验证集选出）": ba, "α 的验证集 Hits@1": bh2,
        "H 单独": h_H, "AN 单独": h_AN, "AV 单独": h_AV,
        "K 单独（纯语义基线）": h_K, "融合": h_F,
        "结构路由净贡献": h_F - h_K,
        "语义路由退化行占比": degen_mean,
        "融合_并列数均值": tie_mean,
        "融合_Hits@1乐观": _hits1_from_ranks(r_F),
    }
    print(f"  ③ H {h_H:.4f} | AN {h_AN:.4f} | AV {h_AV:.4f} | K {h_K:.4f} "
          f"| 融合 {h_F:.4f} (@w={bw:.2f}, α={ba:.2f}) ⇒ 结构净贡献 "
          f"{h_F - h_K:+.4f}", flush=True)

    # ---------------- ④ 免训练方法（统一评测脚本）----------------
    rec["methods"] = {}
    for mname, ctor in (("NNSim", lambda: NNSim(seed=args.seed)),
                        ("AttrSim", lambda: AttrSim(seed=args.seed)),
                        ("StructSim", lambda: StructSim(seed=args.seed, chunk=args.chunk)),
                        ("LinFuse", lambda: LinFuse(d_sem=args.d_sem, seed=args.seed,
                                                    chunk=args.chunk))):
        try:
            obj = ctor().fit(pair)
            res = mcore.evaluate(obj, pair, chunk=args.chunk)
            res.pop("ranks", None)
            rec["methods"][mname] = {k: v for k, v in res.items() if k != "ranks"}
            print(f"  ── {mname:<10} Hits@1={res['Hits@1']:.4f} "
                  f"MRR={res['MRR']:.4f} 并列={res['并列数均值']:,.1f}", flush=True)
        except Exception as exc:                                  # noqa: BLE001
            rec["methods"][mname] = {"error": str(exc)}
            print(f"  ── {mname} 失败：{exc}", flush=True)

    # ---------------- ⑤ CUSSM（训练轮数网格由 CLI 决定）----------------
    try:
        sc = CUSSM(seed=args.seed, prop_chunk=args.chunk,
                  rounds_grid=tuple(args.rounds_grid)).fit(pair)
        res = mcore.evaluate(sc, pair, chunk=args.chunk)
        res.pop("ranks", None)
        rec["CUSSM"] = {k: v for k, v in res.items() if k != "ranks"}
        rec["CUSSM"]["flags"] = sc.flags()
        rec["CUSSM"]["tune_log"] = [list(t) for t in getattr(sc, "tune_log", [])]
        rec["CUSSM"]["融合路数"] = len(sc.fusion_routes())
        print(f"  ── CUSSM       Hits@1={res['Hits@1']:.4f} 纤维保持率="
              f"{res['纤维保持率']} 选中 "
              f"α={sc.alpha:.2f} β={sc.beta:.2f} γ={sc.gamma:.2f} δ={sc.delta:.2f}",
              flush=True)
    except Exception as exc:                                      # noqa: BLE001
        rec["CUSSM"] = {"error": str(exc)}
        print(f"  ── CUSSM 失败：{exc}", flush=True)

    rec["wall_seconds"] = round(time.time() - t0, 1)
    rec["peak_rss_mb"] = round(mcore.peak_rss_mb(), 1)
    print(f"  ── 用时 {rec['wall_seconds']}s，峰值内存 {rec['peak_rss_mb']}MB",
          flush=True)
    return rec


class _FusedView:
    """把"验证集选出的 (w, α) 融合"包装成一个最小 matcher，只为复用统一评测脚本
    算并列诊断。它没有任何可训练参数，也不参与主表。"""

    family = "路由分解"

    def __init__(self, m: CUSSM, sp: StructProp, w: float, a: float):
        self.m, self.sp, self.w, self.a = m, sp, w, a

    def score_block(self, pair, left_idx):
        H = self.sp.score_block(left_idx)
        AN, AV = self.m._k_parts(left_idx)
        K = self.w * AN + (1.0 - self.w) * AV
        return (self.a * _zscore(H) + (1.0 - self.a) * _zscore(K)).astype(np.float32)

    def fiber_labels(self):
        return None


# ---------------------------------------------------------------- 汇总表
def write_markdown(recs: list[dict], args, env: dict) -> str:
    L = ["# 第 5 章跨数据集实测（结构覆盖率与路由判别力）\n",
         f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
         f"- 运行环境：{env['platform']}，Python {env['python']}，numpy {env['numpy']}",
         f"- 命令：`{' '.join(env['argv'])}`",
         f"- 口径：全部 **`argmax`**（模型实际输出，并列按下标小者裁决）；"
         f"`(w, α)` 只在**验证种子**上选，测试集只用于读数",
         f"- 训练轮数网格：{list(args.rounds_grid)}（本脚本不训练 torch 模型）\n"]

    L.append("## 一、结构覆盖率与路由分解\n")
    L.append("| 数据集 | 左/右实体 | 种子/验证/测试 | 结构覆盖率 | 测试查询画像全零 | "
             "H 单独 | AN 单独 | AV 单独 | K 单独 | 融合 | 结构净贡献 | "
             "选中 w | 选中 α | 同名占比 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        if "error" in r:
            L.append(f"| {r['dataset']} | 运行失败：{r.get('error')} |||||||||||||")
            continue
        s, c, t = r["scale"], r["coverage"], r["routes"]
        L.append("| {d} | {el:,}/{er:,} | {ns:,}/{nv:,}/{nt:,} | {cov:.3f} | {tz:.3f} | "
                 "{h:.4f} | {an:.4f} | {av:.4f} | {k:.4f} | **{f:.4f}** | {net:+.4f} | "
                 "{w:.2f} | {a:.2f} | {sn:.3f} |".format(
                     d=r["dataset"], el=s["n_ent_L"], er=s["n_ent_R"],
                     ns=s["n_seeds"], nv=s["n_val"], nt=s["n_test"],
                     cov=c["画像非零的左侧实体占比"], tz=c["测试查询画像全零占比"],
                     h=t["H 单独"], an=t["AN 单独"], av=t["AV 单独"],
                     k=t["K 单独（纯语义基线）"], f=t["融合"],
                     net=t["结构路由净贡献"], w=t["w 字面权重（验证集选出）"],
                     a=t["α 结构权重（验证集选出）"],
                     sn=r["surface"]["测试对中两侧表层名完全相同的占比"]))

    L.append("\n> **读法**。**结构覆盖率**＝结构画像非零的左侧实体占比（＝与至少一个"
             "训练种子相邻）——覆盖不到的查询，结构路由必然失分，故它是结构路由的"
             "**绝对上界**。**K 单独**才是「不用结构路由」的正确基线（α=0 的同一融合"
             "得分），**不是** `max(AN, AV)`：语义路由的两块分别对齐后加权合成，"
             "融合值通常高于任一单块。**结构净贡献**＝融合 − K 单独。\n")

    L.append("\n## 二、表层名分解（同名 vs 异名）\n")
    L.append("| 数据集 | 同名占比 | 同名子集 NNSim Hits@1 | 异名子集 NNSim Hits@1 | "
             "同名查询数 | 异名查询数 |")
    L.append("|---|---|---|---|---|---|")
    for r in recs:
        if "error" in r:
            continue
        sf = r["surface"]
        L.append(f"| {r['dataset']} | {sf['测试对中两侧表层名完全相同的占比']:.4f} | "
                 f"{_f(sf['同名子集_NNSim_Hits@1'])} | "
                 f"{_f(sf['异名子集_NNSim_Hits@1'])} | "
                 f"{sf['同名子集查询数']:,} | {sf['异名子集查询数']:,} |")
    L.append("\n> **为什么必须分解**：两侧表层名完全相同的测试对，表层名相似度方法"
             "近乎白拿；整体成绩若不分解，会被误读为「跨语言匹配成功」。\n")

    L.append("\n## 三、免训练方法的统一评测\n")
    L.append("| 数据集 | 方法 | Hits@1 | Hits@5 | Hits@10 | MRR | MAP@10 | 中位排名 | "
             "Hits@1(乐观) | 并列数均值 | 纤维保持率 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        if "error" in r:
            continue
        for mn in ("NNSim", "AttrSim", "StructSim", "LinFuse", "CUSSM"):
            d = r["methods"].get(mn) if mn != "CUSSM" else r.get("CUSSM")
            if not d or "error" in d:
                L.append(f"| {r['dataset']} | {mn} | 失败 ||||||||||")
                continue
            L.append("| {ds} | {mn} | **{h1:.4f}** | {h5:.4f} | {h10:.4f} | {mrr:.4f} | "
                     "{mp:.4f} | {mr:.0f} | {hop:.4f} | {tie:,.1f} | {fib} |".format(
                         ds=r["dataset"], mn=mn, h1=d["Hits@1"], h5=d["Hits@5"],
                         h10=d["Hits@10"], mrr=d["MRR"], mp=d["MAP@10"],
                         mr=d["MedRank"], hop=d["Hits@1_乐观"],
                         tie=d["并列数均值"], fib=_f(d.get("纤维保持率"))))
    L.append("\n> CUSSM 一栏的传播轮数网格见文件头；torch 训练类基线"
             "（TransE-NN / MTransE / JAPE / 迭代传播）的数值见 "
             "`results/RESULTS.md`（由 `run_experiment.py` 产出）。\n")
    return "\n".join(L) + "\n"


def _f(x):
    return "—" if x is None else f"{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--max-test", type=int, default=1000, dest="max_test",
                    help="限制评测查询数（0 表示不限制）")
    ap.add_argument("--max-images", type=int, default=None, dest="max_images")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--d-sem", type=int, default=256, dest="d_sem")
    ap.add_argument("--rounds-grid", type=int, nargs="*", default=[0],
                    dest="rounds_grid", help="CUSSM 的结构传播轮数网格")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--merge", action="store_true",
                    help="把本次结果**并入**已有的 ch5_measure.json（定向复跑用），"
                         "而不是整份覆盖。没有它时，只跑一个子集会覆盖全部产物。")
    a = ap.parse_args()

    outdir = os.path.join(ROOT, "results")
    os.makedirs(outdir, exist_ok=True)
    env = {"platform": platform.platform(), "python": sys.version.split()[0],
           "numpy": np.__version__, "argv": ["python", "code/measure_ch5.py"] + sys.argv[1:]}
    print("运行环境：", env)

    recs = []
    for k in (a.datasets or ALL_DATASETS):
        try:
            recs.append(measure(k, a))
        except Exception as exc:                                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            recs.append({"dataset": k, "error": str(exc)})

    # ---- 防覆盖护栏 ----
    # 教训（2026-09-28）：曾用 `--datasets flickr30k_entities` 做定向排查，
    # 结果把 8 数据集（42 分钟）的产物**整份覆盖**成只剩 1 条记录。
    # 故：若已有产物、且本次的有效数据集集合是它的**真子集**，先备份再写。
    outp = os.path.join(outdir, "ch5_measure.json")
    if os.path.exists(outp):
        try:
            old = json.load(open(outp, encoding="utf-8"))
            old_ds = {r.get("dataset") for r in old.get("records", [])}
            new_ds = {k for k in (a.datasets or ALL_DATASETS)}
            if old_ds and new_ds < old_ds:
                bak = outp.replace(".json", ".bak.json")
                shutil.copy2(outp, bak)
                print(f"\n[护栏] 本次只跑 {len(new_ds)} 个数据集，是已有产物 "
                      f"({len(old_ds)} 个) 的真子集 → 已备份到 {os.path.basename(bak)}")
        except Exception as exc:                                    # noqa: BLE001
            print(f"[护栏] 备份检查失败（不影响出数）：{exc}")
    # ---- 组装最终记录：--merge 时先读旧产物，再把本次结果并入 ----
    final = _j(recs)
    if getattr(a, "merge", False) and os.path.exists(outp):
        old = json.load(open(outp, encoding="utf-8"))
        by = {r.get("dataset"): r for r in old.get("records", [])}
        for r in final:
            by[r.get("dataset")] = r                    # 本次结果覆盖同名项
        order = [k for k in ALL_DATASETS if k in by] + \
                [k for k in by if k not in ALL_DATASETS]
        final = [by[k] for k in order if k in by]
        print(f"\n[--merge] 本次 {len(recs)} 条并入已有的 {len(old.get('records', []))} 条"
              f" → 共 {len(final)} 条")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump({"env": env, "args": vars(a), "records": final}, f,
                  ensure_ascii=False, indent=2)
    md = write_markdown(recs, a, env)
    with open(os.path.join(outdir, "CH5_MEASURE.md"), "w", encoding="utf-8",
              newline="\r\n") as f:
        f.write(md)
    print(f"\n=== 已写出 {outdir}/ch5_measure.json 与 {outdir}/CH5_MEASURE.md ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
