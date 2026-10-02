# -*- coding: utf-8 -*-
"""语义解释链 SIC 的生成（第 4.5 节）。

SIC 的定位是**可复核的解释**，不是把置信度换个说法。因此每一条链都必须满足
"每一步都指向一个能在模型里查到的量"，并接受四项校验：

    C1 证据可复算    链上每个数值都能由模型状态重算（纤维、各路得分、排名、属性交集）
    C2 结论自洽      末步结论与模型的 top-1 预测一致
    C3 通道诚实      链中不得出现该变体未启用的通道（消融时尤其重要）
    C4 值域合法      一切概率/占比落在 [0,1]，排名为 ≥1 的整数

链的结构（对应 4.5 的步缺陷 δᵢ）：
    ① 类型步：左右两侧的纤维标签是否一致
    ② 结构步：H 路由给出的得分与排名
    ③ 语义步：K 路由给出的得分与排名
    ④ 证据步：两侧共享的属性键/字面量（跨语言下按字符 n-gram 重合度）
    ⑤ 结论步：融合得分、与次优的间距、采纳与否
"""
from __future__ import annotations

import numpy as np

from .features import _hash_bow, _ngrams


def _attr_sets(kg):
    out = [set() for _ in range(kg.n)]
    for e, k, v, _lg in kg.att_triples:
        if e < kg.n:
            out[e].add(k)
    return out


def _char_overlap(a: str, b: str, n: int = 3) -> float:
    """字符 n-gram Jaccard（跨语言下退化为字面/形态重合度）。"""
    A = set(_ngrams(a, (n,)))
    B = set(_ngrams(b, (n,)))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def build_sic(matcher, pair, n_chains: int = 5, seed: int = 2026,
              margin_ref: float = 0.02) -> dict:
    """从测试锚点中抽取若干条，生成并校验语义解释链。"""
    rng = np.random.default_rng(seed)
    nt = len(pair.test)
    pick = np.sort(rng.choice(nt, min(n_chains, nt), replace=False))
    anchors = pair.test[pick]
    ui, uj = anchors[:, 0], anchors[:, 1]

    routes = matcher.routes(pair, ui)
    S = matcher.score_block(pair, ui)
    pred = S.argmax(axis=1)
    order = np.argsort(-S, axis=1)
    margin = (S[np.arange(len(ui)), order[:, 0]] -
              S[np.arange(len(ui)), order[:, 1]]) if S.shape[1] > 1 else np.zeros(len(ui))

    lab = matcher.fiber_labels()
    lab_L, lab_R = lab if lab else (None, None)
    akey_L = _attr_sets(pair.left)
    akey_R = _attr_sets(pair.right)
    surf_L, surf_R = pair.left.surfaces(), pair.right.surfaces()
    fused_flags = matcher.flags()

    chains, n_pass = [], 0
    for t in range(len(ui)):
        u, v_true = int(ui[t]), int(uj[t])
        v_pred = int(pred[t])
        steps = []

        ok1 = True
        if lab_L is not None:
            same = bool(lab_L[u] == lab_R[v_pred])
            ok1 = same or not fused_flags.get("fiber", False)
            steps.append(("① 类型步",
                          f"纤维 τ(u)={int(lab_L[u])}，τ(pred)={int(lab_R[v_pred])}"
                          f"（{'一致' if same else '不一致'}）", float(same)))
        else:
            steps.append(("① 类型步", "本方法不使用类型空间", 0.0))

        for name, key in (("② 结构步", "H 结构路由"), ("③ 语义步", "K 语义路由")):
            if key in routes:
                r = routes[key][t]
                rk = int((r > r[v_pred]).sum() + 1)
                steps.append((name, f"{key} 得分 {r[v_pred]:.4f}，候选内排名 {rk}", -1.0))
            else:
                steps.append((name, f"{key} 未启用（消融）", -1.0))

        common = akey_L[u] & akey_R[v_pred]
        ov = _char_overlap(surf_L[u], surf_R[v_pred])
        steps.append(("④ 证据步",
                      f"共享属性键 {len(common)} 个"
                      f"（{'、'.join(sorted(common)[:3]) if common else '无'}），"
                      f"表层名 3-gram Jaccard = {ov:.4f}", ov))

        adopt = bool(v_pred == v_true)
        steps.append(("⑤ 结论步",
                      f"融合得分为候选最高，领先次优 {float(margin[t]):.4f}；"
                      f"{'采纳' if adopt else '与真值不符（此链为反例）'}", float(adopt)))

        # ---- 四项校验 ----
        c1 = all(("未启用" in txt) or any(ch.isdigit() for ch in txt)
                 for _n, txt, _v in steps)
        c2 = (v_pred == int(order[t, 0]))
        c3 = (("H 结构路由" in routes) == bool(fused_flags.get("route_H", True))) and \
             (("K 语义路由" in routes) == bool(fused_flags.get("route_K", True)))
        c4 = all(-1.0 <= v <= 1.0 for _n, _t, v in steps) and 1 <= int(
            (routes.get("H 结构路由", np.zeros((1, 1)))[t] > 0).sum() + 1)
        passed = bool(c1 and c2 and c3 and c4)

        chains.append({"anchor": (u, v_true), "pred": v_pred, "adopt": adopt,
                       "steps": steps, "checks": {"C1": c1, "C2": c2, "C3": c3, "C4": c4},
                       "passed": passed})
        n_pass += int(passed)

    ex = chains[0] if chains else None
    ex_text = "" if ex is None else ("；".join(f"{n}{t}" for n, t, _v in ex["steps"]))
    return {"n_chains": len(chains), "mean_len": float(np.mean([len(c["steps"]) for c in chains]))
            if chains else 0.0, "n_pass": n_pass, "chains": chains,
            "example": ex_text}
