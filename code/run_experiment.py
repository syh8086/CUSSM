#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一实验入口 —— 第 5 章实验方案的代码对偶。

一条命令跑完"数据集 → 基线 → 本文方法 → 对比 → 指标 → 统计检验"：

    python code/run_experiment.py                      # 全部数据集（默认，跑满轮数）
    python code/run_experiment.py --datasets dbp15k_fr_en --max-test 1000   # 冒烟
    python code/run_experiment.py --methods CUSSM MTransE NNSim
    python code/run_experiment.py --budget-mode wall --max-minutes 3        # 本机限时短跑

**训练预算口径**（`--budget-mode`）—— 两条口径的读数**不可混用**：

    epochs（默认）  跑满 `--epochs` 轮，不设墙上时间上限。腾讯云 GPU 用这个，
                    得到的是「该方法收敛后能到多少」。
    wall            以 `--max-minutes` 为墙上时间上限。本机 CPU 短跑用，得到的是
                    「此预算下能到多少」。被截断的方法会在 stdout 与 RESULTS.md 中
                    标出实际完成轮数，提示不得当作跑满轮数的口径引用。

    同时给出 `--budget-mode epochs` 与 `--max-minutes` 会直接报错（自相矛盾）。

**输入输出一致性的落实**（第 5.4.4 节）：
    输入 —— 所有方法只吃 `cussm.data.load_*` 返回的同一个 `Pair`，特征只由
             `cussm.features.build_view` 产出，没有任何方法自带特征或自带划分；
    输出 —— 所有方法只实现 `score_block(pair, left_idx)`，排名与指标由
             `metrics.core.evaluate` 统一计算，连分块大小都相同。
因此"输入输出一致"不是声明，而是被接口强制的事实。

产物：
    results/results.json          全量结构化结果
    results/ranks/*.npy           逐查询排名（供复算与统计检验）
    results/RESULTS.md            可直接黏进第 5 章的中文表格
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from cussm import data as sdata                     # noqa: E402
from cussm import sps as ssps                       # noqa: E402
from cussm import torch_backend as tb               # noqa: E402
from cussm.model import CUSSM                        # noqa: E402
from cussm.sic import build_sic                     # noqa: E402
from baselines.models import (AttrSim, JAPE, LinFuse, MTransE, NNSim,  # noqa: E402
                              StructPropK, StructSim, TransENN)
from metrics import core as mcore                  # noqa: E402
from metrics import stats as mstats                # noqa: E402

OUT = os.path.join(ROOT, "results")
RANKDIR = os.path.join(OUT, "ranks")

# 全部数据集（第 5.1 节的 8 个方向 / 6 类公开数据源）
ALL_DATASETS = ["dbp15k_fr_en", "dbp15k_zh_en", "dbp15k_ja_en",
                "flickr30k_entities", "countries_s1",
                "fb15k237", "wn18", "yago3_10"]


def _epochs_done_of(m) -> object:
    """取模型实际完成的训练轮数。

    两种形态：独立训练（TransE-NN）两侧各记一个数 ⇒ `list`；
    联合训练（MTransE/JAPE 及 CUSSM 的结构路由）记总轮数 ⇒ `int`。
    """
    v = getattr(m, "epochs_done", None)
    if v is None:
        v = (getattr(getattr(m, "emb", None), "meta", None) or {}).get("epochs_done")
    return v


def _is_truncated(done, planned: int) -> bool:
    """训练是否**未跑满规划轮数**（即被 `--max-minutes` 截断）。

    判定取严格口径：任一计数小于规划轮数即算截断 —— 因为被截断的读数与跑满的读数
    **不可混用**（前者是"这个预算下能到多少"，后者是"这个方法收敛后能到多少"），
    哪怕只差 1 轮也应标出来，免得审稿人替我们发现问题。
    """
    if done is None:
        return False
    vals = done if isinstance(done, (list, tuple)) else [done]
    return any(int(v) < planned for v in vals)


def _assert_same_fibers(m, pair, args) -> None:
    """核对 CUSSM 自己用的类型骨架与基类绑定的是**同一套**。

    纤维保持率 `ρ_fib`（4.3.3）只有在"所有方法用同一个 `π_A, π_B`"时才可跨行比较。
    基类 `bind_fibers` 与 `CUSSM.fit` 走同一个 `_fiber_labels_of` 缓存，故**必须**逐位
    相同；一旦不等，说明参数（`d_sem/n_types/seed`）不一致，此时表 12 该列不可比 ——
    这种情况按项目规则**报错停机**，不静默继续。
    """
    if not (getattr(m, "use_fiber", False) or getattr(m, "use_type", False)):
        return                                    # 该变体本就没构造骨架（标签为占位零）
    own_L, own_R = getattr(m, "lab_L", None), getattr(m, "lab_R", None)
    bound = getattr(m, "_fib", None)
    if own_L is None or bound is None:
        return
    if not (np.array_equal(own_L, bound[0]) and np.array_equal(own_R, bound[1])):
        raise RuntimeError(
            "CUSSM 的纤维标签与 `bind_fibers` 绑定的一致 —— 若不一致，基线报告的 "
            "ρ_fib 与本文方法用的不是同一套类型骨架，跨行比较不成立。"
            "请检查 d_sem/n_types/seed 是否与 args 一致。")


def make_methods(args) -> dict:
    """全部调用者共享同一组训练超参（`--dim/--epochs/--n-neg/--lr/--device`），
    因此基线之间、基线与本文方法之间的差别**只来自方法本身**。

    CUSSM 只接收它真正使用的超参：结构嵌入族的 (dim/epochs/...) 通过 `**kw` 透传即可，
    但不与显式参数重复（曾因重复传 `seed` 抛 `TypeError`）。
    """
    ep = args.epochs
    emb = dict(dim=args.dim, epochs=ep, n_neg=args.n_neg, lr=args.lr,
               batch=args.batch, seed=args.seed, device=args.device,
               max_minutes=args.max_minutes)
    methods = {
        "NNSim": lambda: NNSim(seed=args.seed),
        "AttrSim": lambda: AttrSim(seed=args.seed),
        "StructSim": lambda: StructSim(seed=args.seed, chunk=args.chunk),
        "StructProp-自训练": lambda: StructPropK(seed=args.seed, chunk=args.chunk),
        "LinFuse": lambda: LinFuse(d_sem=args.d_sem, seed=args.seed, chunk=args.chunk),
        "TransE-NN": lambda: TransENN(**emb),
        "MTransE": lambda: MTransE(**emb),
        "JAPE": lambda: JAPE(**emb),
        "CUSSM": lambda: CUSSM(d_sem=args.d_sem, n_types=args.n_types,
                             prop_chunk=args.chunk, seed=args.seed,
                             rounds_grid=((0,) if args.quick else (0, 1, 2)),
                             grid_delta=((0.0,) if args.quick
                                         else (0.0, 0.05, 0.15, 0.3))),
    }
    # ---- B 轨：近五年结构侧方法（2026-09-30）----
    # 方案 `.workbuddy/repro_plan_trackB_2026-09-30.md`（路线乙：按原文算法在本文契约下
    # 重实现）。容错导入 —— 该包缺失时主链照常，不因新增基线影响既有实验。
    try:
        from baselines.recent import RECENT_BASELINES
    except Exception as exc:                       # noqa: BLE001
        print(f"  （B 轨近五年基线未注册：{type(exc).__name__}: {exc}）", flush=True)
        RECENT_BASELINES = {}
    for _key, _cls in RECENT_BASELINES.items():
        # 编码器隐层取 2×结构嵌入维度（--dim 128 → 256），与原文量级相当；
        # 其余超参与结构嵌入族**同源**（同 epochs/lr/batch/seed/device/预算）。
        methods[_key] = (lambda c=_cls: c(
            d=args.dim * 2, n_layer=2, epochs=args.epochs, lr=args.lr,
            batch=args.batch, d_sem=args.d_sem, seed=args.seed,
            device=args.device, max_minutes=args.max_minutes))
    return methods


# ---- 「通道强制开启」消融组（2026-09-29 新增）----
# 动机：通道准入（`CUSSM._need`）在若干数据集上把 β、γ、δ **全部**判为"不采纳"，
# 于是「−纤维约束 / −类型通道 / −粘合通道」三行与主表**逐位相同**——它们是空操作，
# 无法回答"这三条通道有没有用"。根因是：门槛回答的是"该不该采纳"，
# 而论文要回答的是"启用后能带来多少增益"，两者不是同一个问题。
# 本组把 `channels_must_on=True` 打开（取消门槛、在**非零**网格内取验证集 argmax，
# 保证 β、γ、δ 均 >0），再逐条关掉一条，从而分离出每条通道的**边际增益**；
# 若某条的增益为负，那就是该通道在本数据上的真实代价，照实报。
FORCE_ABLATIONS = {
    # 注意：名字会进入 ranks/*.npy 的**文件名**，故**禁用** Windows 非法字符
    # `< > : " / \ | ? *`。曾用「（β,γ,δ>0）」导致 `Permission denied` 落盘失败。
    "CUSSM + 通道强制开启（β,γ,δ 均非零）": dict(channels_must_on=True),
    "CUSSM + 强制开启 − 纤维罚（β=0）": dict(channels_must_on=True, use_fiber=False),
    "CUSSM + 强制开启 − 类型先验（γ=0）": dict(channels_must_on=True, use_type=False),
    "CUSSM + 强制开启 − 粘合通道（δ=0）": dict(channels_must_on=True, use_glue=False),
}


def make_ablations(args) -> dict:
    """消融只改**一个**构造开关，其余与主表 CUSSM 逐位相同（走同一份代码路径）。

    分两组，由 `--ablation-set` 选择：

    **standard（常规 6 组）**——"关掉某个构件会怎样"。
    **force（强制开启 4 组）**——"把通道强行打开能拿到多少增益"。
    """
    standard = {
        "CUSSM − 纤维约束": dict(use_fiber=False),
        "CUSSM − 类型通道": dict(use_type=False),
        "CUSSM − 粘合通道": dict(use_glue=False),
        "CUSSM − 结构路由H": dict(use_route_H=False),
        "CUSSM − 语义路由K": dict(use_route_K=False),
        "CUSSM − 超参选择（w=α=0.5，β=γ=δ=0）": dict(tune=False),
    }
    which = getattr(args, "ablation_set", "standard")
    if which == "force":
        return dict(FORCE_ABLATIONS)
    if which == "both":
        return {**standard, **FORCE_ABLATIONS}
    return standard


def _attach_mm_encoder(pair, args, name: str) -> dict:
    """Track A′ 专用：训练跨模态编码器并把嵌入回填为两侧的语义视图。

    为什么要接在 `run_dataset` 里而不是另开入口：项目纪律要求"主表与消融跑**同一份
    代码路径**"。把编码器当作**语义视图的来源**接进来，则 NNSim / LinFuse / CUSSM /
    消融全部自动看到同一份嵌入，方法之间的差别仍然只来自方法本身。

    编码器只读 `pair.seeds` 做监督（`train_two_stage` 内部保证不读 `pair.test`）。
    """
    from cussm import mm_encoder as mme
    if not mme.has_deps():
        raise RuntimeError(
            "缺少跨模态编码器依赖（transformers / Pillow / torch）。"
            "装法：pip install transformers pillow")
    t0 = time.time()
    print(f"  ── 训练跨模态编码器（Track A′）"
          f"{'：区域上限 ' + str(args.mm_limit_regions) if args.mm_limit_regions else ''}")
    zL, zR, rep = mme.train_two_stage(
        pair, epochs1=args.mm_epochs1, epochs2=args.mm_epochs2, d=args.mm_d,
        bs=args.mm_batch, bs2=args.mm_bs2, workers=args.mm_workers,
        seed=args.seed, device=None if args.device == "auto" else args.device,
        limit_regions=args.mm_limit_regions, stage2_pairs=args.mm_stage2_pairs,
        verbose=True)
    mme.attach(pair, zL, zR,
               note=f"跨模态编码器（ViT-B/16 + BERT-base，d={args.mm_d}，"
                    f"两阶段 epochs={args.mm_epochs1}/{args.mm_epochs2}）")
    rep = rep.as_dict()
    rep["wall_seconds"] = round(time.time() - t0, 1)
    # 编码器自身在**测试集**上的检索读数：它是"结构层的对照"，不是本文方法的成绩，
    # 故单列而非混进 methods 表。
    rep["encoder_only_retrieval"] = mme.evaluate_retrieval(
        zL, zR, pair.test, chunk=args.chunk)
    print(f"     编码器自身测试集 R@1="
          f"{rep['encoder_only_retrieval']['region→phrase']['R@1']:.4f}"
          f"  R@10={rep['encoder_only_retrieval']['region→phrase']['R@10']:.4f}"
          f"  （候选池 {pair.right.n}，随机 R@1={rep['encoder_only_retrieval']['random_R@1']:.2e}）"
          f"  用时 {rep['wall_seconds']}s")
    return rep


def run_dataset(name: str, args) -> dict:
    print(f"\n{'=' * 78}\n### 数据集 {name}\n{'=' * 78}")
    t_start = time.time()
    ep = args.epochs                      # 规划轮数，用于识别训练是否被截断
    kw = {}
    if args.max_test:
        kw["max_test"] = args.max_test
    if name.startswith("flickr") and args.max_images:
        kw["max_images"] = args.max_images
    pair = sdata.LOADERS[name](seed=args.seed, **kw)
    print("  " + pair.summary())

    mm_rec = None
    if getattr(args, "mm_encoder", False):
        if not name.startswith("flickr"):
            print(f"  （--mm-encoder 只对 flickr 轨道生效，{name} 跳过）")
        else:
            try:
                mm_rec = _attach_mm_encoder(pair, args, name)
            except Exception as exc:                   # noqa: BLE001
                print(f"  ✗ 编码器训练失败：{exc}")
                traceback.print_exc()
                mm_rec = {"error": str(exc)}

    methods = make_methods(args)
    if args.methods:
        keep = set(args.methods)
        methods = {k: v for k, v in methods.items() if k in keep}
    ablations = make_ablations(args) if args.ablations else {}

    rec = {"dataset": name, "pair": pair.summary(), "meta": pair.meta,
           "methods": {}, "ablations": {}, "alpha": None, "sic": None,
           "mm_encoder": mm_rec}

    # ---------------- 1) 逐方法拟合与评测 ----------------
    base_before = mcore.peak_rss_mb()
    fitted = {}
    for mname, ctor in methods.items():
        print(f"  ── {mname}", flush=True)
        try:
            m = ctor().fit(pair)
            # 绑定统一类型骨架（4.1 定义 4）：使 9 个方法的 ρ_fib 落在**同一把尺子**上。
            # 此前只有 CUSSM 返回标签、基类一律 `None` ⇒ 主表该列对 8 个基线全空；
            # 而该指标在定义上对**任何给出预测的方法**都成立（见 5.3.2）。
            m.bind_fibers(pair, d_sem=args.d_sem, n_types=args.n_types, seed=args.seed)
            if mname == "CUSSM":
                _assert_same_fibers(m, pair, args)
            res = mcore.evaluate(m, pair, chunk=args.chunk)
            hit1 = mstats.hit_vector(res["ranks"], 1)
            lo, hi, _ = mstats.bootstrap_ci(hit1, B=args.boot, seed=args.seed)
            np.save(os.path.join(RANKDIR, f"{name}__{mname.replace(' ', '_')}.npy"),
                    res.pop("ranks"))
            ed = _epochs_done_of(m)
            rec["methods"][mname] = {
                "family": m.family, "flags": m.flags(),
                "Hits@1": res["Hits@1"], "Hits@5": res["Hits@5"], "Hits@10": res["Hits@10"],
                "MRR": res["MRR"], "MAP@10": res["MAP@10"], "MedRank": res["MedRank"],
                "纤维保持率": res["纤维保持率"], "n_query": res["n_query"],
                "Hits@1_CI": [lo, hi],
                # 并列诊断：主指标与乐观上界同时入表，供审稿人核对
                "Hits@1_乐观": res["Hits@1_乐观"], "MRR_乐观": res["MRR_乐观"],
                "并列数均值": res["并列数均值"], "含并列行占比": res["含并列行占比"],
                "fit_seconds": round(getattr(m, "train_time", 0.0), 2),
                # 训练预算口径入档：`--budget-mode wall` 下结构嵌入族会被 `--max-minutes`
                # 截断，实际完成轮数必须留痕，否则"截断读数"会被当成"收敛读数"引用。
                # 独立训练（TransE-NN）两侧各记一个数；联合训练（MTransE/JAPE）与
                # CUSSM 的结构路由记总轮数（落在 `JointEmb.meta["epochs_done"]`）。
                "epochs_done": ed,
                "epochs_planned": ep,
                "epochs_truncated": _is_truncated(ed, ep),
                "score_seconds": res["score_seconds"],
                "params": m.n_params(),
                "embed_params": (m.n_embed_params() if hasattr(m, "n_embed_params")
                                 else m.param_detail().get("结构嵌入表 E_L,E_R", 0)),
                "param_detail": m.param_detail(),
            }
            fitted[mname] = m
            print(f"     Hits@1={res['Hits@1']:.4f}  MRR={res['MRR']:.4f}  "
                  f"并列={res['并列数均值']:,.0f}  "
                  f"Hits@1(乐观)={res['Hits@1_乐观']:.4f}  "
                  f"纤维保持率={_f(res['纤维保持率'])}")
            if rec["methods"][mname]["epochs_truncated"]:
                print(f"     ⚠ 训练被截断：实际完成 {ed} / 规划 {ep} 轮 —— "
                      f"该行**不得**当作跑满轮数的口径引用")
        except Exception as exc:                       # noqa: BLE001
            print(f"     ✗ 失败：{exc}")
            traceback.print_exc()
            rec["methods"][mname] = {"error": str(exc)}

    # ---------------- 2) 先取三分量，再批量标定 α，最后定 Δ 与 SPS ----------------
    # 三分量与 α 无关，故先以占位 α=(0,0,0) 取出分量，再用**全体方法**标定 α，
    # 避免"参考方法某分量为 0 ⇒ 该分量对所有方法失效"（见 sps.calibrate_alpha_multi）。
    comps = {}
    n_offered = {}
    for mname, m in fitted.items():
        if "error" in rec["methods"].get(mname, {}):
            continue
        try:
            au = ssps.audit(m, pair, n_sample=args.audit_n, seed=args.seed,
                            alpha=(0.0, 0.0, 0.0))
            rec["methods"][mname].update({k: au[k] for k in
                                          ("Δ_comm", "Δ_comm_floor", "Δ_nat", "Δ_glue",
                                           "Δ_glue_cocone", "Δ_glue_pool",
                                           "n_route", "n_route_offered",
                                           "fusion_routes", "routes")})
            comps[mname] = [au["Δ_comm"], au["Δ_nat"], au["Δ_glue"]]
            # 填补池的判据是"模型是否**具备**多路由"（n_route_offered），而不是
            # "融合是否**实际启用**多路由"（n_route）。后者在"备了多路但验证集把权重
            # 压到一路"的数据集上会让池整体为空、填补静默失效（见 sps 文档 2026-09-29 修）。
            n_offered[mname] = int(au["n_route_offered"])
        except Exception as exc:                       # noqa: BLE001
            print(f"     ✗ {mname} 的 SPS 分量核算失败：{exc}")
            rec["methods"][mname]["sps_error"] = str(exc)

    # α 用**未填补**的分量标定 —— 量纲应来自真实测量，而不是填补值。
    ref = "CUSSM" if "CUSSM" in comps else (list(comps)[0] if comps else None)
    alpha = ssps.calibrate_alpha_multi(comps, ref) if comps else (0.0, 0.0, 0.0)
    rec["alpha"] = [round(a, 4) for a in alpha]
    # 口径标记：本产物由"填补池按 n_route_offered 判定"的代码产出 ⇒ 离线重算脚本
    # （.workbuddy/resps_offered.py）据此跳过，避免对已是新口径的产物重复施加。
    rec["sps_pool"] = "offered"
    print(f"  ── α 批量标定（参考 {ref}，零分量退全体中位数）: α={rec['alpha']}")

    # 填补未具备多路由的方法所缺失的 Δ_nat / Δ_glue（否则 SPS 奖励"少做事"，见 sps 文档）。
    comps, imputed = ssps.impute_missing_components(comps, n_offered,
                                                     rule=args.sps_impute)
    if imputed:
        rec["sps_imputed"] = {k: v for k, v in imputed.items()}
        for k, js in imputed.items():
            print(f"     · {k} 未具备多路由，Δ_{'nat' if 1 in js else ''}"
                  f"{'/' if len(js) > 1 else ''}{'glue' if 2 in js else ''}"
                  f" 按全体具备多路由的方法的最大值填补")

    for mname, c in comps.items():
        d, s = ssps.aggregate(c, alpha)
        rec["methods"][mname]["Δ_nat"] = c[1]
        rec["methods"][mname]["Δ_glue"] = c[2]
        rec["methods"][mname]["Δ"] = d
        rec["methods"][mname]["SPS"] = s
        rec["methods"][mname]["alpha"] = rec["alpha"]
        rec["methods"][mname]["Δ_nat_imputed"] = 1 in imputed.get(mname, [])
        rec["methods"][mname]["Δ_glue_imputed"] = 2 in imputed.get(mname, [])
        print(f"     {mname:<14} SPS={s:.4f}  Δ={d:.4f}  "
              f"(Δ_comm={_f(c[0])} Δ_nat={_f(c[1])} Δ_glue={_f(c[2])})")

    # ---------------- 3) 消融 ----------------
    abl_comp, abl_n_offered = {}, {}
    for aname, kw in ablations.items():
        print(f"  ── {aname}", flush=True)
        try:
            m = CUSSM(seed=args.seed, **kw).fit(pair)
            res = mcore.evaluate(m, pair, chunk=args.chunk)
            np.save(os.path.join(RANKDIR, f"{name}__{aname.replace(' ', '_')}.npy"),
                    res.pop("ranks"))
            au = ssps.audit(m, pair, n_sample=args.audit_n, seed=args.seed, alpha=alpha)
            rec["ablations"][aname] = {
                "flags": m.flags(), "Hits@1": res["Hits@1"], "MRR": res["MRR"],
                "Hits@10": res["Hits@10"], "纤维保持率": res["纤维保持率"],
                "Hits@1_乐观": res["Hits@1_乐观"], "并列数均值": res["并列数均值"],
                "Δ_comm": au["Δ_comm"], "Δ_nat": au["Δ_nat"], "Δ_glue": au["Δ_glue"],
                "n_route": au["n_route"], "n_route_offered": au["n_route_offered"],
                "fit_seconds": round(getattr(m, "train_time", 0.0), 2),
                # 与主表同一口径：消融变体若被截断，同样不得当作收敛读数引用。
                "epochs_done": _epochs_done_of(m),
                "epochs_planned": ep,
                "epochs_truncated": _is_truncated(_epochs_done_of(m), ep),
                "params": m.n_params(),
            }
            abl_comp[aname] = [au["Δ_comm"], au["Δ_nat"], au["Δ_glue"]]
            abl_n_offered[aname] = int(au["n_route_offered"])
            print(f"     Hits@1={res['Hits@1']:.4f}  并列={res['并列数均值']:,.0f}  "
                  f"纤维保持率={res['纤维保持率']}  路由数={au['n_route']}")
            if rec["ablations"][aname]["epochs_truncated"]:
                print("     ⚠ 训练被截断，该行不得当作跑满轮数的口径引用")
        except Exception as exc:                       # noqa: BLE001
            print(f"     ✗ 失败：{exc}")
            rec["ablations"][aname] = {"error": str(exc)}

    # 消融里同样有未具备多路由的变体（如关掉一整条路由），一并按同一口径填补。
    if abl_comp:
        abl_comp, abl_imp = ssps.impute_missing_components(
            abl_comp, abl_n_offered, rule=args.sps_impute)
        for aname, c in abl_comp.items():
            d, s = ssps.aggregate(c, alpha)
            rec["ablations"][aname].update({"Δ": d, "SPS": s,
                                            "Δ_nat": c[1], "Δ_glue": c[2],
                                            "imputed": abl_imp.get(aname, [])})
            print(f"     {aname:<26} SPS={s:.4f}  Δ={d:.4f}"
                  + ("  [Δ_nat/Δ_glue 已填补]" if aname in abl_imp else ""))

    # ---------------- 4) 统计检验（本文方法 vs 各基线）----------------
    if "CUSSM" in rec["methods"] and "error" not in rec["methods"]["CUSSM"]:
        s_r = np.load(os.path.join(RANKDIR, f"{name}__CUSSM.npy"))
        s_h = mstats.hit_vector(s_r, 1)
        pv, nm = [], []
        for mname in methods:
            if mname == "CUSSM" or "error" in rec["methods"].get(mname, {}):
                continue
            b_h = mstats.hit_vector(
                np.load(os.path.join(RANKDIR, f"{name}__{mname.replace(' ','_')}.npy")), 1)
            t = mstats.paired_permutation(s_h, b_h, B=args.perm, seed=args.seed)
            pv.append(t["p"]); nm.append(mname)
            rec["methods"][mname]["vs_CUSSM"] = t
        rec["holm"] = mstats.holm(pv, nm) if pv else []
        if pv:
            print("  ── Holm 校正：")
            for row in rec["holm"]:
                print(f"     {row['对照']:<14} p={row['p']:.4g}  "
                      f"阈值={row['阈值']:.4g}  {'显著' if row['显著'] else '不显著'}")

    # ---------------- 5) 纤维约束的 (精度, 约束落实率) 权衡前沿 ----------------
    # 4.2 的纤维保持是**硬约束**：它的产出是"约束被真正落实"（纤维保持率），
    # 而不是 Hits@1。把硬约束塞进精度目标里，它一旦净亏就会被关掉，论文承诺的
    # 约束便形同虚设。故主表仍取**精度优先**（保证 CUSSM ≥ LinFuse），
    # 另用本节把 β 扫出一条 Pareto 前沿，供 4.2 引用。
    # 实测（DBP15K zh_en）：(β,γ)=(0.10,0.15) 在纤维保持率 +4.9 点的同时
    # Hits@1 还 +0.006 —— 严格 Pareto 改善；(β,γ)=(1.00,0.15) 换 +22.9 点保持率
    # 的代价是 Hits@1 −0.036。
    if "CUSSM" in fitted and fitted["CUSSM"].use_fiber:
        try:
            m = fitted["CUSSM"]
            beta_used = float(m.beta)                  # 调参选中的值，扫完必须复位
            lab_L, lab_R = m.lab_L, m.lab_R
            li = pair.test[:, 0]
            gt = pair.test[:, 1]
            fr = []
            for b in sorted({0.0, 0.05, 0.10, 0.25, 0.50, 1.00, beta_used}):
                m.beta = float(b)
                pred = []
                for i in range(0, len(li), args.chunk):
                    blk = li[i:i + args.chunk]
                    pred.append(np.argmax(m.score_block(pair, blk), axis=1))
                pred = np.concatenate(pred)
                fr.append({"β": float(b),
                           "Hits@1": float((pred == gt).mean()),
                           "纤维保持率": float((lab_L[li] == lab_R[pred]).mean())})
            m.beta = beta_used
            rec["纤维约束前沿"] = fr
            print("  ── 纤维约束前沿（β → 精度 / 约束落实率）：")
            for row in fr:
                print(f"     β={row['β']:.2f}  Hits@1={row['Hits@1']:.4f}  "
                      f"纤维保持率={row['纤维保持率']:.4f}")
        except Exception as exc:                       # noqa: BLE001
            print(f"     ✗ 前沿核算失败：{exc}")
            rec["纤维约束前沿"] = {"error": str(exc)}

    # ---------------- 6) SIC 示范 ----------------
    if "CUSSM" in fitted:
        try:
            rec["sic"] = build_sic(fitted["CUSSM"], pair, seed=args.seed)
            print(f"  ── SIC 示范：{rec['sic']['n_chains']} 条链，"
                  f"平均长度 {rec['sic']['mean_len']:.2f}，"
                  f"通过校验 {rec['sic']['n_pass']} 条")
        except Exception as exc:                       # noqa: BLE001
            print(f"     ✗ SIC 生成失败：{exc}")
            rec["sic"] = {"error": str(exc)}

    rec["peak_rss_mb"] = round(mcore.peak_rss_mb(), 1)
    rec["rss_delta_mb"] = round(mcore.peak_rss_mb() - base_before, 1)
    rec["wall_seconds"] = round(time.time() - t_start, 1)
    print(f"  ── 数据集完成，用时 {rec['wall_seconds']}s，峰值内存 {rec['peak_rss_mb']}MB")
    return rec


def _f(x):
    return "—" if x is None else f"{x:.4f}"


# ---------------------------------------------------------------- 汇总表
def write_markdown(all_rec: list[dict], args, env: dict) -> str:
    L = []
    L.append("# 第 5 章 实验结果（本地 / 云端复现）\n")
    L.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- 运行环境：Python {env.get('python') or sys.version.split()[0]}，"
             f"numpy {env.get('numpy') or np.__version__}，scipy {env.get('scipy')}，"
             f"torch {env.get('torch')}")
    L.append(f"- 平台与主机：{env.get('platform') or sys.platform}，"
             f"{env.get('cpu_count')} vCPU，主机名 `unknown`")
    L.append(f"- 训练设备：**{env.get('device_name')}**（`{env.get('device')}`，"
             f"CUDA {env.get('cuda')}）")
    L.append(f"- 训练超参：dim={args.dim} epochs={args.epochs} n_neg={args.n_neg} "
             f"lr={args.lr} batch={args.batch} λ_align={args.lam_align}；"
             f"随机种子 {args.seed}；bootstrap B={args.boot}；置换检验 B={args.perm}\n")
    # 预算口径必须明写：等轮数 vs 等墙上时间，两者的读数不可混用
    if getattr(args, "budget_mode", "epochs") == "wall":
        L.append(f"- 预算口径：**等墙上时间**（单模型上限 {args.max_minutes} 分钟）"
                 "—— 受机器速度影响的「此预算下能到多少」口径\n")
    else:
        L.append("- 预算口径：**等轮数**（每个方法跑满 `epochs`，不设墙上时间上限）"
                 "—— 与机器速度无关的「收敛后能到多少」口径\n")
    L.append("> **数据来源**：所有数据集由 `code/download_data.py` 从公开镜像下载，"
             "逐文件以 GitHub Contents API 的 git blob SHA-1 校验；来源清单见 "
             "`data/manifest.json`。**全部数值为实跑结果，非文献引用值、非预估。**\n")

    for rec in all_rec:
        L.append(f"\n## {rec['dataset']}\n")
        if "error" in rec:
            L.append(f"运行失败：{rec['error']}\n")
            continue
        L.append(f"{rec['pair']}\n")
        L.append(f"α（量纲归一，参考方法 CUSSM）= {rec['alpha']}\n")
        L.append("| 方法 | 类别 | 路由数 | Hits@1 | 95% CI | Hits@5 | Hits@10 | MRR | MAP@10 | "
                 "中位排名 | Hits@1(乐观) | 并列数均值 | 纤维保持率 | SPS | Δ | 映射参数 | "
                 "嵌入表参数 | 训练(s) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        any_imp = False
        for m, d in rec["methods"].items():
            if "error" in d:
                L.append(f"| {m} | — | | 失败：{d['error'][:40]} | | | | | | | | | | | | | | |")
                continue
            ci = (f"[{d['Hits@1_CI'][0]:.4f}, {d['Hits@1_CI'][1]:.4f}]"
                  if d.get("Hits@1_CI") else "—")
            im = (d.get("Δ_nat_imputed") or d.get("Δ_glue_imputed"))
            any_imp = any_imp or bool(im)
            L.append("| {m} | {fam} | {nr} | **{h1:.4f}** | {ci} | {h5:.4f} | {h10:.4f} | "
                     "{mrr:.4f} | {mp:.4f} | {mr:.0f} | {hop:.4f} | {tie:,.0f} | {fib} | "
                     "{sps}{mk} | {dl} | {p} | {ep} | {t} |".format(
                         m=m, fam=d.get("family", "—"), nr=d.get("n_route", "—"),
                         h1=d["Hits@1"], ci=ci,
                         h5=d["Hits@5"], h10=d["Hits@10"], mrr=d["MRR"], mp=d["MAP@10"],
                         mr=d["MedRank"], hop=d.get("Hits@1_乐观", float("nan")),
                         tie=d.get("并列数均值", float("nan")),
                         fib=_f(d.get("纤维保持率")),
                         sps=_f(d.get("SPS")), mk="†" if im else "", dl=_f(d.get("Δ")),
                         p=f"{d['params']:,}", ep=f"{d.get('embed_params', 0):,}",
                         t=d["fit_seconds"]))
        L.append("\n> **Hits@1 为模型实际输出口径**（并列按下标小者裁决，与 `argmax` 逐位"
                 "一致）：`Hits@1 = 1` 当且仅当模型发出的那一个预测就是真值。"
                 "「Hits@1(乐观)」把并列一律记为 rank 1（部分文献口径），"
                 "「并列数均值」给出真值得分并列的候选数 —— 两者并列看才知道"
                 "高 Hits@1 是否来自并列。结构路由的得分是整数计数，并列可达数万。\n")
        if any_imp:
            L.append("\n> † 该方法的 Δ_nat / Δ_glue 为**单路由**（无跨路由变换、无融合），"
                     "定义上不计；若不填补则 SPS 会奖励「少做事」（实测 NNSim 反超 "
                     "LinFuse）。入表前按同一数据集内**所有多路由方法的对应分量"
                     "最大值**填补。\n")
        trunc = [(mn, d) for mn, d in rec["methods"].items() if d.get("epochs_truncated")]
        if trunc:
            L.append("\n> ⚠ **训练被截断**：" + "；".join(
                f"`{mn}` 实际完成 {d.get('epochs_done')} / 规划 {d.get('epochs_planned')} 轮"
                for mn, d in trunc) +
                " —— 由 `--max-minutes`（`--budget-mode wall`）提前收尾，"
                "该口径是「此预算下能到多少」而非「收敛后能到多少」，"
                "**不得**当作跑满轮数的结果引用；云端跑满轮数后须整行替换。\n")
        if rec.get("ablations"):
            L.append(f"\n### {rec['dataset']} — 消融\n")
            L.append("| 变体 | Hits@1 | Hits@1(乐观) | 并列数均值 | Hits@10 | MRR | 纤维保持率 | "
                     "SPS | Δ | Δ_comm | Δ_nat | Δ_glue |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for a, d in rec["ablations"].items():
                if "error" in d:
                    L.append(f"| {a} | 失败 | | | | | | | | | | |")
                    continue
                L.append("| {a} | {h1:.4f} | {hop:.4f} | {tie:,.0f} | {h10:.4f} | {mrr:.4f} | "
                         "{fib} | {sps} | {dl} | {dc} | {dn} | {dg} |".format(
                             a=a, h1=d["Hits@1"], hop=d.get("Hits@1_乐观", float("nan")),
                             tie=d.get("并列数均值", float("nan")),
                             h10=d["Hits@10"], mrr=d["MRR"],
                             fib=_f(d["纤维保持率"]), sps=_f(d["SPS"]), dl=_f(d["Δ"]),
                             dc=_f(d["Δ_comm"]), dn=_f(d["Δ_nat"]), dg=_f(d["Δ_glue"])))
            atr = [an for an, d in rec["ablations"].items() if d.get("epochs_truncated")]
            if atr:
                L.append("\n> ⚠ 消融中被截断的变体：" + "、".join(f"`{an}`" for an in atr)
                         + "（同主表口径，不得当作跑满轮数的结果引用）。\n")
        if rec.get("holm"):
            L.append(f"\n### {rec['dataset']} — CUSSM vs 基线（配对置换检验，Holm 校正）\n")
            # 符号口径：正值 = **本文方法更优**。原来渲染的是 `-diff`（对照 − 本文），
            # 而列名只写「差异(Hits@1)」⇒ 读起来与直觉相反；第 5 章表 12b 用
            # 「本文 − 对照」，两处必须同向，否则同一批数字会给人相反的印象。
            L.append("| 对照 | 对照 Hits@1 | ΔHits@1（本文 − 对照） | p 值 | Holm 阈值 | 显著 |")
            L.append("|---|---|---|---|---|---|")
            for row in rec["holm"]:
                mv = rec["methods"].get(row["对照"], {}) or {}
                diff = (mv.get("vs_CUSSM") or {}).get("diff", float("nan"))
                h1 = mv.get("Hits@1", float("nan"))
                L.append(f"| {row['对照']} | {h1:.4f} | {diff:+.4f} | {row['p']:.4g} | "
                         f"{row['阈值']:.4g} | {'✓' if row['显著'] else '✗'} |")
        if rec.get("纤维约束前沿") and "error" not in rec["纤维约束前沿"]:
            L.append(f"\n### {rec['dataset']} — 纤维约束的 (精度, 约束落实率) 前沿\n")
            L.append("β（纤维罚强度）固定其余超参后扫描；**硬约束的产出是约束落实率，"
                     "不是 Hits@1**，故主表取精度优先，前沿单独列出：\n")
            L.append("| β | Hits@1 | 纤维保持率 |")
            L.append("|---|---|---|")
            for row in rec["纤维约束前沿"]:
                L.append(f"| {row['β']:.2f} | {row['Hits@1']:.4f} | {row['纤维保持率']:.4f} |")
        if rec.get("sic") and "error" not in rec["sic"]:
            L.append(f"\n### {rec['dataset']} — SIC 示范\n")
            L.append(f"生成 {rec['sic']['n_chains']} 条链，平均长度 "
                     f"{rec['sic']['mean_len']:.2f}，通过四类校验 "
                     f"{rec['sic']['n_pass']}/{rec['sic']['n_chains']}。"
                     f"示例：{rec['sic']['example']}\n")
    md = "\n".join(L) + "\n"
    return md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="默认全部 8 个：" + " ".join(ALL_DATASETS))
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--max-test", type=int, default=None, help="限制评测查询数（调试）")
    ap.add_argument("--max-images", type=int, default=None, help="Flickr 图像数上限（调试）")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--audit-n", type=int, default=400, help="SPS 审计采样锚点数")
    ap.add_argument("--sps-impute", default="max", choices=["max", "median"],
                    dest="sps_impute",
                    help="单路由方法缺失的 Δ_nat/Δ_glue 如何填补（max=最保守，默认）")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    # ---- 训练超参：全部方法共用同一组，保证公平 ----
    ap.add_argument("--dim", type=int, default=128, help="结构嵌入维度")
    ap.add_argument("--epochs", type=int, default=60, help="联合训练轮数")
    ap.add_argument("--n-neg", type=int, default=4, dest="n_neg", help="TransE 负例数")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lam-align", type=float, default=1.0, dest="lam_align")
    ap.add_argument("--d-sem", type=int, default=256, dest="d_sem", help="语义视图维度")
    ap.add_argument("--n-types", type=int, default=16, dest="n_types", help="类型纤维数")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto：有 CUDA 就用 GPU（腾讯云 GPU 实例用 auto 即可）")
    # ---- Track A′：跨模态编码器（第 5.4 节「端到端训练环节」的入口）----
    ap.add_argument("--mm-encoder", dest="mm_encoder", action="store_true",
                    help="对 flickr 轨道训练双塔跨模态编码器（ViT-B/16 + BERT-base）"
                         "并把嵌入回填为两侧的语义视图；其余数据集自动跳过")
    ap.add_argument("--mm-epochs1", type=int, default=6, dest="mm_epochs1",
                    help="阶段一（线性探测）轮数")
    ap.add_argument("--mm-epochs2", type=int, default=1, dest="mm_epochs2",
                    help="阶段二（端到端微调）轮数；0 表示只做线性探测")
    ap.add_argument("--mm-d", type=int, default=256, dest="mm_d",
                    help="两塔共用的嵌入维度")
    ap.add_argument("--mm-batch", type=int, default=64, dest="mm_batch",
                    help="阶段一取图像的批大小")
    ap.add_argument("--mm-bs2", type=int, default=16, dest="mm_bs2",
                    help="阶段二每批的图文对数")
    ap.add_argument("--mm-workers", type=int, default=4, dest="mm_workers",
                    help="取图像的数据加载进程数")
    ap.add_argument("--mm-limit-regions", type=int, default=None, dest="mm_limit_regions",
                    help="只编码前 N 个区域（冒烟用）；未编码区域以零向量占位并在结果中标注")
    ap.add_argument("--mm-stage2-pairs", type=int, default=20000, dest="mm_stage2_pairs",
                    help="阶段二用的种子对数上限（骨干前向是瓶颈）")
    ap.add_argument("--budget-mode", default="epochs", choices=["epochs", "wall"],
                    dest="budget_mode",
                    help="训练预算口径（默认 epochs）。epochs：跑满 --epochs 轮、不设墙上"
                         "时间上限 —— 腾讯云 GPU 用这个，得到「收敛后能到多少」；"
                         "wall：以 --max-minutes 为墙上时间上限 —— 本机 CPU 短跑用，"
                         "得到「此预算下能到多少」。两种口径的读数不可混用，"
                         "同时给出会直接报错。")
    ap.add_argument("--max-minutes", type=float, default=None, dest="max_minutes",
                    help="训练墙上时间上限，**仅 --budget-mode wall 下生效**")
    ap.add_argument("--quick", action="store_true",
                    help="冒烟模式：缩小轮数与网格（本机 CPU 上用）")
    ap.add_argument("--no-ablations", dest="ablations", action="store_false", default=True)
    ap.add_argument("--ablation-set", choices=("standard", "force", "both"),
                    default="standard",
                    help="消融组：standard=常规 6 组（−通道/−路由/−调参）；"
                         "force=通道强制开启 4 组（绕开显著性门槛，度量通道的边际增益）；"
                         "both=全部 10 组。force/both 会额外拟合 CUSSM，但**不改**任何既有读数。")
    ap.add_argument("--out", default=None,
                    help="结果 JSON 的输出路径（默认 results/results.json）；同名 .md 一并写出。"
                         "供冒烟/矩阵自检使用，避免覆盖主表产物（2026-09-28 教训）。")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：若 `--out` 指定的 JSON 已存在，跳过其中**已成功落盘**的"
                         "数据集，只补跑缺失的（2026-09-28 教训：云端长跑在最后一个数据集被"
                         "OOM 杀掉时，原实现只在全部跑完后写一次盘，导致前 6 个数据集约 7 小时"
                         "的计算全部作废）。")
    a = ap.parse_args()

    # ---- 预算口径的互斥校验：混用会把"截断读数"混进"收敛读数" ----
    if a.budget_mode == "epochs":
        if a.max_minutes is not None:
            ap.error("--budget-mode epochs（默认，跑满轮数）与 --max-minutes（墙上时间"
                     "上限）互斥：前者要求跑满 --epochs 轮，后者会在中途截断。"
                     "若确实要限时短跑，请显式写 --budget-mode wall。")
        a.max_minutes = None          # 显式入档：本轮预算口径是「轮数」而非「时间」
    elif a.max_minutes is None:
        ap.error("--budget-mode wall 必须同时给出 --max-minutes，否则与 epochs 模式等价、"
                 "口径无从区分。")

    if a.quick:
        a.epochs = min(a.epochs, 8)
        a.perm = min(a.perm, 2000)
        a.boot = min(a.boot, 500)
        a.audit_n = min(a.audit_n, 120)

    os.makedirs(RANKDIR, exist_ok=True)
    env = tb.device_report(a.device)
    print("运行环境：", env)
    names = a.datasets or ALL_DATASETS
    all_rec = []

    # 输出路径可覆盖：矩阵/冒烟自检用 `--out` 写到独立文件，避免把主表产物覆盖成
    # 单数据集的中间结果（2026-09-28 实测踩过：矩阵档运行期间重建第 5 章读到中间态）。
    outp = a.out or os.path.join(OUT, "results.json")
    outm = (outp[:-5] if outp.endswith(".json") else outp) + ".md"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)

    def _flush() -> None:
        """把当前已完成的记录**立刻落盘**（原子替换，避免读到写了一半的 JSON）。

        2026-09-28 教训：原实现把 `all_rec` 攒在内存里、只在全部数据集跑完后写一次，
        云端第 7 个数据集（yago3_10）CUSSM 阶段 OOM 被杀时，前 6 个数据集约 7 小时的
        计算**全部作废**。改为每完成一个数据集即落盘后，任何时刻中断最多只损失当前
        数据集。
        """
        payload = {"args": vars(a), "env": env, "records": all_rec}
        tmp = outp + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, outp)
        with open(outm, "w", encoding="utf-8", newline="\r\n") as f:
            f.write(write_markdown(all_rec, a, env))

    # ---- 断点续跑：跳过 `--out` 里已成功落盘的数据集 ----
    done: dict = {}
    if a.resume and os.path.exists(outp):
        try:
            with open(outp, encoding="utf-8") as f:
                prev = json.load(f)
            for r in prev.get("records", []):
                m = r.get("methods") or {}
                if r.get("dataset") and m and "error" not in r \
                        and not any("error" in v for v in m.values()):
                    done[r["dataset"]] = r
            print(f"[resume] {outp} 中已成功落盘 {len(done)} 个数据集，将跳过：{sorted(done)}")
        except Exception as exc:                        # noqa: BLE001
            print(f"[resume] 读取 {outp} 失败，忽略并从头跑：{exc}")

    for n in names:
        if n in done:
            print(f"\n[resume] 跳过已完成数据集 {n}")
            all_rec.append(done[n])
            continue
        try:
            rec = run_dataset(n, a)
        except Exception as exc:                        # noqa: BLE001
            print(f"✗ {n} 整体失败：{exc}")
            traceback.print_exc()
            rec = {"dataset": n, "error": str(exc), "methods": {}, "ablations": {}}
        all_rec.append(rec)
        _flush()                                        # ← 每完成一个数据集立刻落盘
        print(f"\n=== 已增量写出 {outp}（{len(all_rec)}/{len(names)} 个数据集）===")

    _flush()
    print(f"\n=== 已写出 {outp} 与 {outm} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
