#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""纤维硬约束的 **(精度, 约束落实率) 权衡前沿** —— 第 4.2 节的验收口径。

## 为什么单独有这个脚本

4.2 的纤维保持 $\pi_B\circ F=\pi_A$ 是**硬约束**：它的产出是"约束被真正落到解里"
（纤维保持率 $\rho_{\text{fib}}$），而不是 Hits@1。把硬约束塞进精度目标里考核，
它一旦净亏就会被关掉，论文承诺的约束便形同虚设。因此第 5 章的**主表取精度优先**
（保证 CUSSM 在关闭全部附加通道时不劣于两视图线性融合的对照），
而纤维罚 $\beta$ 与类型先验 $\gamma$ 的价值另用本脚本扫出的**前沿**来验收。

`run_experiment.py` 的第 5 节只扫 $\beta$（一维，其余超参固定在调参选中的值）。
本脚本把 $(\beta,\gamma)$ 做成**二维网格**并同时报 $\delta$ 的增量，因此：

- 能读出**严格 Pareto 改善点**（精度与约束落实率同时变好），这是"硬约束可免费落实"
  的直接证据；
- 能读出**代价曲线**（用多少精度换多少落实率），供 4.2 讨论取舍；
- 能读出**粘合通道的增量**（第 3 节），这是 4.4 负面结论的出处。

## 用法

    python code/measure_frontier.py --datasets dbp15k_fr_en dbp15k_zh_en wn18
    python code/measure_frontier.py --datasets dbp15k_fr_en --max-test 1000

    # 云端分片（一次只跑一个方向）：逐次并入同一个产物，分片之间不会互相覆盖
    python code/measure_frontier.py --datasets dbp15k_fr_en --out results/ch5_frontier.json
    python code/measure_frontier.py --datasets wn18 --out results/ch5_frontier.json --merge

产物：默认 `results/ch5_frontier.json`、`results/CH5_FRONTIER.md`；
       `--out` 改路径（同名 `.md` 一并写出），`--merge` 按数据集并入已有文件
       （同数据集以本次为准），供云端把多个分片合到一份。

指标一律取 `argmax` 口径（与主表一致）。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from metrics import core as mcore                    # noqa: E402
from cussm import data as sdata                       # noqa: E402
from cussm.model import CUSSM                          # noqa: E402

_ACCEPTS_MAX_TEST = {"dbp15k_fr_en", "dbp15k_zh_en", "dbp15k_ja_en",
                     "fb15k237", "wn18", "yago3_10"}

BETAS = (0.0, 0.05, 0.10, 0.25, 0.50, 1.00)
GAMMAS = (0.0, 0.05, 0.15, 0.40)
DELTAS = (0.0, 0.05, 0.15, 0.30)


def _j(o):
    if isinstance(o, dict):
        return {k: _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_j(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return _j(o.tolist())
    return o


_CROSS_CHECKED = {"done": False}


def _measure(m: CUSSM, pair, li, gt, chunk: int, cross_check: bool = True) -> dict:
    """在给定超参下的 (Hits@1, 纤维保持率)。

    裁决口径与主表**同源**：`Hits@1 = (argmax(S) == gt)`，等价于
    `metrics.core.ranks(tie="consistent") <= 1`（5.3.1）。本函数为省时间只打一次
    分就同时算出两项，故不能直接调 `evaluate`（后者会为并列诊断再扫一遍）；
    作为补偿，**首次调用时用统一评测代码交叉验算**，口径不一致即报错 ——
    这样"评测代码唯一"（5.4.4）是可被脚本自身证明的，而不是口头承诺。
    """
    lab_L, lab_R = m.lab_L, m.lab_R
    hit, keep = 0, 0
    for i in range(0, len(li), chunk):
        blk = li[i:i + chunk]
        pred = np.argmax(m.score_block(pair, blk), axis=1)
        hit += int((pred == gt[i:i + chunk]).sum())
        keep += int((lab_L[blk] == lab_R[pred]).sum())
    n = len(li)
    if cross_check and n and not _CROSS_CHECKED["done"]:
        k = min(64, n)
        r = mcore.ranks(m, pair, li[:k], gt[:k], chunk=chunk)
        h_ref = float((r <= 1).mean())
        h_here = float((np.argmax(m.score_block(pair, li[:k]), axis=1) == gt[:k]).mean())
        print(f"  [交叉验算] 统一评测 mcore.ranks 的 Hits@1={h_ref:.6f}，"
              f"本脚本 argmax 口径={h_here:.6f} ⇒ "
              f"{'一致' if abs(h_ref - h_here) <= 1e-12 else '**不一致，中止**'}",
              flush=True)
        if abs(h_ref - h_here) > 1e-12:
            raise AssertionError(
                f"评测口径不一致：本脚本 {h_here:.6f} vs metrics.core {h_ref:.6f}")
        _CROSS_CHECKED["done"] = True
    return {"Hits@1": hit / n, "纤维保持率": keep / n}


def run(key: str, args) -> dict:
    t0 = time.time()
    print(f"\n{'=' * 78}\n### {key}\n{'=' * 78}", flush=True)
    kw = {"seed": args.seed}
    if args.max_test and key in _ACCEPTS_MAX_TEST:
        kw["max_test"] = args.max_test
    pair = sdata.LOADERS[key](**kw)
    print("  " + pair.summary(), flush=True)

    li, gt = pair.test[:, 0], pair.test[:, 1]
    m = CUSSM(seed=args.seed, prop_chunk=args.chunk,
             rounds_grid=tuple(args.rounds_grid)).fit(pair)
    tuned = {"α": float(m.alpha), "β": float(m.beta), "γ": float(m.gamma),
             "δ": float(m.delta), "w": float(m.w),
             "val_hits@1": float(getattr(m, "val_hits1", float("nan")))}
    base = _measure(m, pair, li, gt, args.chunk)
    print(f"  调参选中：α={m.alpha:.2f} β={m.beta:.2f} γ={m.gamma:.2f} δ={m.delta:.2f}"
          f" ⇒ Hits@1={base['Hits@1']:.4f} 纤维保持率={base['纤维保持率']:.4f}",
          flush=True)

    # 复原函数：把 (α, β, γ, δ) 写回模型，测完必须复位
    a0, b0, g0, d0 = m.alpha, m.beta, m.gamma, m.delta

    # ---------------- ① β × γ 二维前沿 ----------------
    grid = []
    for b in BETAS:
        for g in GAMMAS:
            m.beta, m.gamma = float(b), float(g)
            r = _measure(m, pair, li, gt, args.chunk)
            r.update({"β": float(b), "γ": float(g)})
            grid.append(r)
    m.beta, m.gamma = b0, g0

    # ---------------- ② δ 增量（在调参选中的 α 下）----------------
    dgrid = []
    for d in sorted({*DELTAS, d0}):
        m.delta = float(d)
        r = _measure(m, pair, li, gt, args.chunk)
        r["δ"] = float(d)
        dgrid.append(r)
    m.delta = d0

    # ---------------- ③ Pareto 判定（相对 β=γ=0 的基线）----------------
    ref = next(r for r in grid if r["β"] == 0.0 and r["γ"] == 0.0)
    pareto = [r for r in grid
              if r["Hits@1"] >= ref["Hits@1"] - 1e-12
              and r["纤维保持率"] > ref["纤维保持率"] + 1e-12]
    strict = [r for r in grid
              if r["Hits@1"] > ref["Hits@1"] + 1e-12
              and r["纤维保持率"] > ref["纤维保持率"] + 1e-12]

    print(f"  ① β×γ 网格 {len(grid)} 格；β=γ=0 基线 Hits@1={ref['Hits@1']:.4f} "
          f"保持率={ref['纤维保持率']:.4f}", flush=True)
    print(f"  ② 零精度代价可提升保持率的格：{len(pareto)} 个；"
          f"严格 Pareto 改善（精度与保持率同时升）：{len(strict)} 个", flush=True)
    for r in strict:
        print(f"     β={r['β']:.2f} γ={r['γ']:.2f} ⇒ Hits@1 {r['Hits@1']:.4f} "
              f"({r['Hits@1'] - ref['Hits@1']:+.4f})，保持率 "
              f"{r['纤维保持率']:.4f} ({r['纤维保持率'] - ref['纤维保持率']:+.4f})",
              flush=True)

    # ---------------- ④ 复位后复核（防止扫描污染主表）----------------
    m.alpha, m.beta, m.gamma, m.delta = a0, b0, g0, d0
    check = _measure(m, pair, li, gt, args.chunk)
    ok = (abs(check["Hits@1"] - base["Hits@1"]) < 1e-12
          and abs(check["纤维保持率"] - base["纤维保持率"]) < 1e-12)
    print(f"  ③ 复位复核：{'通过' if ok else '**不通过**'}"
          f"（{check['Hits@1']:.4f} / {check['纤维保持率']:.4f}）", flush=True)

    return {"dataset": key, "pair": pair.summary(),
            "n_query": int(len(li)), "candidates": int(pair.right.n),
            "tuned": tuned, "baseline": base,
            "beta_gamma_grid": grid, "delta_grid": dgrid,
            "zero_cost_improve": pareto, "strict_pareto": strict,
            "reset_ok": bool(ok),
            "beta_grid": list(BETAS), "gamma_grid": list(GAMMAS),
            "delta_scan": list(sorted({*DELTAS, d0})),
            "wall_seconds": round(time.time() - t0, 1)}


def write_markdown(recs: list[dict], args, env: dict) -> str:
    L = ["# 纤维硬约束的 (精度, 约束落实率) 前沿\n",
         f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
         f"- 运行环境：{env['platform']}，Python {env['python']}，numpy {env['numpy']}",
         f"- 命令：`{' '.join(env['argv'])}`",
         f"- 口径：`argmax`（模型实际输出）；纤维保持率 = 预测对落在同一纤维的占比；"
         f"传播轮数网格 {list(args.rounds_grid)}\n",
         "> **为什么要这条前沿**：纤维保持是**硬约束**（4.2），其产出是「约束被真正"
         "落实」而不是精度。主表取精度优先（保证附加通道只可能加分），约束的落实程度"
         "在此单列验收。\n"]

    for r in recs:
        if "error" in r:
            L.append(f"\n## {r['dataset']}\n\n运行失败：{r['error']}\n")
            continue
        t = r["tuned"]
        L.append(f"\n## {r['dataset']}\n")
        L.append(f"查询 {r['n_query']:,}，候选 {r['candidates']:,}；"
                 f"调参选中 α={t['α']:.2f}、β={t['β']:.2f}、γ={t['γ']:.2f}、"
                 f"δ={t['δ']:.2f}、w={t['w']:.2f}"
                 f"（验证集 Hits@1={t['val_hits@1']:.4f}）；"
                 f"复位复核 {'通过' if r['reset_ok'] else '**不通过**'}\n")
        ref = next(x for x in r["beta_gamma_grid"]
                   if x["β"] == 0.0 and x["γ"] == 0.0)
        L.append(f"**β=γ=0 基线**：Hits@1 = {ref['Hits@1']:.4f}，"
                 f"纤维保持率 = {ref['纤维保持率']:.4f}\n")
        L.append("| β | γ | Hits@1 | ΔHits@1 | 纤维保持率 | Δ保持率 | 判定 |")
        L.append("|---|---|---|---|---|---|---|")
        for x in r["beta_gamma_grid"]:
            dh = x["Hits@1"] - ref["Hits@1"]
            dk = x["纤维保持率"] - ref["纤维保持率"]
            tag = ("严格 Pareto 改善" if (dh > 1e-12 and dk > 1e-12)
                   else "零精度代价" if (abs(dh) <= 1e-12 and dk > 1e-12)
                   else "以精度换保持率" if dk > 1e-12
                   else "—")
            L.append(f"| {x['β']:.2f} | {x['γ']:.2f} | {x['Hits@1']:.4f} | "
                     f"{dh:+.4f} | {x['纤维保持率']:.4f} | {dk:+.4f} | {tag} |")
        L.append("\n**δ（粘合通道）单独扫描**\n")
        L.append("| δ | Hits@1 | ΔHits@1 | 纤维保持率 |")
        L.append("|---|---|---|---|")
        for x in r["delta_grid"]:
            L.append(f"| {x['δ']:.2f} | {x['Hits@1']:.4f} | "
                     f"{x['Hits@1'] - ref['Hits@1']:+.4f} | {x['纤维保持率']:.4f} |")
        L.append("")
    return "\n".join(L) + "\n"


def merge_records(prev: dict, recs: list[dict]) -> list[dict]:
    """把本次结果按 `dataset` 并入已有产物（同数据集以**本次**为准）。

    保持已有记录的**原有顺序**，本次新增的数据集追加到末尾 —— 这样"逐方向分片跑"
    合出来的记录顺序与"一次跑完"一致，便于人工比对。
    若本次某方向失败而旧记录是成功的，会打印警告（失败记录同样覆盖，如实反映现状）。
    """
    order = [r["dataset"] for r in prev.get("records", []) if r.get("dataset")]
    by = {r["dataset"]: r for r in prev.get("records", []) if r.get("dataset")}
    for r in recs:
        k = r.get("dataset")
        if k in by and "error" in r and "error" not in by[k]:
            print(f"⚠ --merge：本次 `{k}` 失败，将覆盖已有的成功记录")
        if k not in order:
            order.append(k)
        by[k] = r
    return [by[k] for k in order]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["dbp15k_fr_en", "dbp15k_zh_en", "wn18"])
    ap.add_argument("--max-test", type=int, default=1000, dest="max_test")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--rounds-grid", type=int, nargs="*", default=[0],
                    dest="rounds_grid")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None,
                    help="输出 JSON 路径（默认 results/ch5_frontier.json）；同名 .md 一并写出。")
    ap.add_argument("--merge", action="store_true",
                    help="与 --out 指向的已有文件**按数据集合并**（同数据集以本次为准），"
                         "而不是整体覆盖。云端逐方向分片跑时用它把各片并到一份。")
    a = ap.parse_args()

    outdir = os.path.join(ROOT, "results")
    os.makedirs(outdir, exist_ok=True)
    env = {"platform": platform.platform(), "python": sys.version.split()[0],
           "numpy": np.__version__,
           "argv": ["python", "code/measure_frontier.py"] + sys.argv[1:]}
    print("运行环境：", env)

    recs = []
    for k in a.datasets:
        try:
            recs.append(run(k, a))
        except Exception as exc:                                   # noqa: BLE001
            import traceback
            traceback.print_exc()
            recs.append({"dataset": k, "error": str(exc)})

    recs = _j(recs)
    outp = a.out or os.path.join(outdir, "ch5_frontier.json")
    outm = (outp[:-5] if outp.endswith(".json") else outp) + ".md"
    os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)

    if a.merge and os.path.exists(outp):
        try:
            prev = json.load(open(outp, encoding="utf-8"))
        except Exception as exc:                                   # noqa: BLE001
            print(f"✗ --merge 读取 {outp} 失败（{exc}），本次改为整体覆盖")
            prev = {"records": []}
        print(f"--merge：已有 {len(prev.get('records', []))} 条 + 本次 {len(recs)} 条")
        recs = merge_records(prev, recs)
        print(f" ⇒ 合并后 {len(recs)} 条")

    with open(outp, "w", encoding="utf-8") as f:
        json.dump({"env": env, "args": vars(a), "records": recs}, f,
                  ensure_ascii=False, indent=2)
    with open(outm, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(write_markdown(recs, a, env))
    print(f"\n=== 已写出 {outp} 与 {outm} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
