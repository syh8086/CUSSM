# -*- coding: utf-8 -*-
"""SPS 三分量的可执行核算（第 4.3 节）。

聚合式严格照 4.3.1：
    Δ = α₁Δ_comm + α₂Δ_nat + α₃Δ_glue,   SPS = exp(−βΔ) ∈ (0,1],   β = 1

**α 的由来**：4.3.1 的具例取 α=(1.00, 2.15, 0.93)，其构造是"量纲归一"——使三分量
对 Δ 的贡献各占 1/3。验算：(1/0.1163):(1/0.0541):(1/0.1255) = 8.60:18.48:7.97，
归一即 (1.00, 2.15, 0.93)。故本文件不硬编码该三元组，而是提供一个 `calibrate_alpha`
按同一规则在**指定的参考方法**上标定，再把标定结果**冻结**并施加于所有方法
（否则每个方法各自归一就等于取消了量纲归一的意义）。

三分量的可执行定义（每条都给出量纲与参照下限，避免"整数量级由实现决定"）：

Δ_comm  交换性残余：沿左侧关系边 u→u′，其像与 u 的匹配对象 v 的关系边是否重合。
        原始残余 = P(pred(u′) ∉ N(v)∪{v})；再减去**数据自身**的非保持率
        （用真值匹配算出的同一量），即"超出数据固有下限的过额失配"。
        右侧图无关系边时该分量未定义（Flickr Track A′ 即如此），此时返回 None。

Δ_nat   自然性偏离：两条路由在同一候选集上的行分布之总变差距离的一半。
        单路由方法该分量未定义、`audit` 返回 0，**但入表前必须经
        `impute_missing_components` 填补**（否则 SPS 会奖励"少做事"，见该函数）。

Δ_glue  粘合偏离，取 4.4 的两个子分量之均值：
          ① 余锥交换：两模块对**共享锚点**（真值对齐对）的读取之差；
          ② 上界偏移：融合结果与"逐块并置"Pool=(S_H+S_K)/2 的相对 Frobenius 距离。
        无融合（单路由）时同样未定义、返回 0，入表前同法填补。
"""
from __future__ import annotations

import numpy as np

BETA = 1.0          # 4.3.1 具例取 β=1
W_SUM = 3.0         # 归一目标：三分量各占 1/3


# ---------------------------------------------------------------- 工具
def _softmax_rows(S: np.ndarray, T: float | None = None) -> np.ndarray:
    """按行做温度缩放 softmax；T=None 时用行标准差的倒数做尺度无关归一。"""
    S = np.asarray(S, dtype=np.float64)
    if T is None:
        sd = S.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        Z = S / sd
    else:
        Z = S / T
    Z -= Z.max(axis=1, keepdims=True)
    np.exp(Z, out=Z)
    Z /= Z.sum(axis=1, keepdims=True)
    return Z


def tv(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """逐行总变差距离的一半（取值 [0,1]）。"""
    return 0.5 * np.abs(p - q).sum(axis=1)


def calibrate_alpha(d_comm: float | None, d_nat: float, d_glue: float) -> tuple:
    """**单方法**标定：按"三分量各占 1/3"给出 α；未定义或为零的分量取 α=0。

    注意：该形式在参考方法某分量为零时会把整项权重压成 0，从而对**所有方法**
    都消掉该分量。批量比较请改用 `calibrate_alpha_multi`。
    """
    comps = [d_comm if d_comm is not None else 0.0, d_nat, d_glue]
    a = [0.0, 0.0, 0.0]
    for i, d in enumerate(comps):
        if d and d > 0:
            a[i] = (1.0 / W_SUM) / d
    return tuple(a)


def calibrate_alpha_multi(raw: dict[str, list[float | None]], ref: str) -> tuple:
    """**批量**标定：以参考方法的分量定标；参考分量为零/未定义时退到全体中位数。

    `raw` 为 {方法名 → [Δ_comm, Δ_nat, Δ_glue]}（未定义记 None）。
    这样即使参考方法在某分量上恰好为 0（例如完美匹配时 Δ_comm=0），
    该分量对其余方法仍然保有非零权重，不会因为参考值而整体失效。
    """
    ref_v = raw.get(ref) or [None, None, None]
    alpha = []
    for i in range(3):
        vals = sorted(v[i] for v in raw.values() if v[i] is not None and v[i] > 1e-9)
        scale = ref_v[i] if (ref_v[i] is not None and ref_v[i] > 1e-9) else \
            (np.median(vals) if vals else 0.0)
        alpha.append(float((1.0 / W_SUM) / scale) if scale > 1e-9 else 0.0)
    return tuple(alpha)


def aggregate(comp: list[float | None], alpha) -> tuple:
    """由三分量按给定 α 聚合出 (Δ, SPS)。"""
    d = sum(a * (c if c else 0.0) for a, c in zip(alpha, comp))
    return float(d), float(np.exp(-BETA * d))


def impute_missing_components(comps: dict, n_offered: dict, rule: str = "max"):
    """**未具备多路由通道的方法之缺失分量填补**（必须做，否则 SPS 会奖励"少做事"）。

    只有一条路由的方法没有跨路由变换，Δ_nat 在定义上取 0；没有融合，Δ_glue 也取 0。
    若照单全收，"不建第二条路由"就成了 SPS 上的净收益。实测（DBP15K fr_en，1500 查询）：

        NNSim   Δ_comm=0.5356  Δ_nat=0.0000  Δ_glue=0.0000  →  SPS=0.6929
        LinFuse Δ_comm=0.3767  Δ_nat=0.8487  Δ_glue=0.2133  →  SPS=0.4572
        CUSSM    Δ_comm=0.4867  Δ_nat=0.8487  Δ_glue=0.3715  →  SPS=0.3679

    SPS 的排序与 Hits@1（LinFuse 0.9180 > CUSSM 0.8653 > NNSim 0.7580）**完全相反**。

    填补口径（保守）：把 `n_offered < 2` 的方法在第 2、3 分量上置为该数据集内
    **所有具备多路由的方法（`n_offered >= 2`）对应分量的最大值**，即"未提供该性质的证据 ⇒ 按已观测到的最不利
    情形计"。这样单路由方法既不因缺通道而获利，也不被罚到比任何实测值更差。

    **池判据为何用"具备"而非"启用"（2026-09-29 修）**：原判据按 `n_route`（**实际进入
    融合**的路由数）筛填补源，在"模型具备多路由、但验证集把第二条路由的权重压到 0"的
    数据集上会整体失效。实测 `countries_s1`：9 个方法的 `n_route` 全为 1（而 LinFuse /
    CUSSM 的 `n_route_offered` 为 2 / 3），池空 ⇒ 填补静默跳过 ⇒ "不建第二条路由"重新成为
    SPS 上的净收益 —— 恰是本函数要修的缺陷复发（NNSim 因此得 `Δ=0 ⇒ SPS=exp(0)=1.0000`
    的退化满分）。判据改为 `n_offered` 后，池由"模型是否**具备**该通道"决定，与超参是否
    选中无关；`n_route_offered` 已由 `audit` 返回并写入产物，故历史产物可离线重算。

    `rule`：`"max"`（默认，最保守）取最大值；`"median"` 取中位数；`"worst"` 与 max 同义。
    返回 `(填补后的 comps, 被填补的方法→被填补的分量下标列表)`。
    """
    comps = {k: list(v) for k, v in comps.items()}
    capable = [k for k in comps if int(n_offered.get(k, 1)) >= 2]
    noted: dict[str, list[int]] = {}
    for j in (1, 2):                      # 只填补 Δ_nat 与 Δ_glue
        vals = [comps[k][j] for k in capable
                if comps[k][j] is not None and comps[k][j] > 1e-12]
        if not vals:
            continue
        fill = max(vals) if rule in ("max", "worst") else float(np.median(vals))
        for k, v in comps.items():
            if int(n_offered.get(k, 1)) >= 2:
                continue
            if v[j] is None or v[j] <= 1e-12:
                v[j] = float(fill)
                noted.setdefault(k, []).append(j)
    return comps, noted


# ---------------------------------------------------------------- 邻接表
def _adj(kg, kind: str = "out") -> dict:
    d: dict[int, set] = {}
    tri = kg.rel_triples
    src, dst = (0, 2) if kind == "out" else (2, 0)
    for i in range(len(tri)):
        d.setdefault(int(tri[i, src]), set()).add(int(tri[i, dst]))
    return d


# ---------------------------------------------------------------- 主审计
def audit(matcher, pair, n_sample: int = 400, seed: int = 2026,
          alpha: tuple | None = None, chunk: int = 512) -> dict:
    """核算一个已拟合方法的 SPS 三分量。"""
    rng = np.random.default_rng(seed)
    n_t = len(pair.test)
    idx = np.sort(rng.choice(n_t, min(n_sample, n_t), replace=False))
    anchors = pair.test[idx]
    ui, uj = anchors[:, 0], anchors[:, 1]

    routes = matcher.routes(pair, ui)
    S_fused = matcher.score_block(pair, ui)
    rnames = list(routes.keys())
    # "逐块并置"的基准只取**真正进入融合**的路由：被提供但权重为零的路由不该出现，
    # 否则模型会因为"备了一条没用上的路由"而被扣分（见 BaseMatcher.fusion_routes）。
    # 实测缺陷：δ=0 时 CUSSM 的 Δ_glue 被算成 0.6448 而 LinFuse 只有 0.2292，
    # 差异**全部**来自那条权重为零的粘合路由 —— 而两者的融合得分逐位相同。
    used = matcher.fusion_routes() if hasattr(matcher, "fusion_routes") else None
    ukeys = [k for k in rnames if used is None or k in used]

    # ---------------- Δ_nat ----------------
    # 自然性是"两路由之间那个变换"的性质，与当前权重无关，故仍取前两条路由。
    if len(rnames) >= 2:
        r1, r2 = routes[rnames[0]], routes[rnames[1]]
        p1, p2 = _softmax_rows(r1), _softmax_rows(r2)
        d_nat = float(tv(p1, p2).mean())
    else:
        d_nat = 0.0

    # ---------------- Δ_glue ----------------
    term1 = term2 = None
    if len(ukeys) >= 2:
        # ① 余锥交换：两模块对共享锚点的读取之差（候选集 = 共享锚点的真值集合）
        blk = [routes[k][:, uj] for k in ukeys]
        ps = [_softmax_rows(b) for b in blk]
        term1 = float(tv(ps[0], ps[1]).mean())
        # ② 上界偏移：融合结果与"逐块并置"Pool 的**方向**偏离。
        # 用 0.5(1−cos) 而不是 ‖Δ‖/‖Pool‖：后者在融合得分被 z-score 标准化后
        # 量纲与原始读数不同，比值可以远超 1（实测 LinFuse 上达到 6.26），
        # 无法与 Δ_nat ∈[0,1] 并列。余弦式定义在任意正尺度下不变且落在 [0,1]。
        pool = np.mean(np.stack(blk), axis=0)
        fused = S_fused[:, uj]
        a_ = np.asarray(pool, np.float64).ravel()
        b_ = np.asarray(fused, np.float64).ravel()
        na, nb = np.linalg.norm(a_), np.linalg.norm(b_)
        term2 = float(0.5 * (1.0 - float(a_ @ b_) / (na * nb))) if na > 0 and nb > 0 else 0.0
        d_glue = (term1 + term2) / 2.0
    else:
        d_glue = 0.0

    # ---------------- Δ_comm ----------------
    d_comm, floor = _commutativity(matcher, pair, anchors, rng, n_sample)

    # ---------------- 聚合 ----------------
    if alpha is None:
        alpha = calibrate_alpha(d_comm, d_nat, d_glue)
    parts = [d_comm if d_comm is not None else 0.0, d_nat, d_glue]
    wsum = sum(a * p for a, p in zip(alpha, parts))
    sps = float(np.exp(-BETA * wsum))
    return {"Δ_comm": d_comm, "Δ_comm_floor": floor, "Δ_nat": d_nat, "Δ_glue": d_glue,
            "Δ_glue_cocone": term1, "Δ_glue_pool": term2,
            "alpha": tuple(round(a, 4) for a in alpha),
            "Δ": float(wsum), "SPS": sps,
            # `n_route` 取**真正进入融合**的路由数 —— 单路由方法（含"提供了多路但权重
            # 全压在一路上"的情形）的 Δ_nat/Δ_glue 未定义，入表前须经
            # `impute_missing_components` 填补，故这里必须是"实际用到的路数"。
            "n_route": len(ukeys), "n_route_offered": len(rnames),
            "fusion_routes": ukeys,
            "routes": rnames, "n_anchor": int(len(anchors))}


def _commutativity(matcher, pair, anchors, rng, n_sample) -> tuple:
    """交换性残余与其数据固有下限。返回 (Δ_comm 或 None, 下限 或 None)。"""
    R = pair.right
    if len(R.rel_triples) == 0:
        return None, None
    outL = _adj(pair.left, "out")
    adjR = _adj(R, "out")
    if not outL or not adjR:
        return None, None

    truth_l = {int(a): int(b) for a, b in pair.test}
    truth_l.update({int(a): int(b) for a, b in pair.val})
    truth_l.update({int(a): int(b) for a, b in pair.seeds})

    # 采样有出边且真值已知的锚点
    pick = []
    for u, v in anchors:
        nb = outL.get(int(u))
        if not nb:
            continue
        pick.append((int(u), int(v), next(iter(nb))))
    if not pick:
        return None, None
    us = np.array([p[0] for p in pick], dtype=np.int64)
    vs = np.array([p[1] for p in pick], dtype=np.int64)
    ups = np.array([p[2] for p in pick], dtype=np.int64)

    pred = matcher.score_block(pair, ups).argmax(axis=1)
    near = np.array([pred[k] in adjR.get(int(vs[k]), ()) or pred[k] == vs[k]
                     for k in range(len(pick))])
    resid_model = 1.0 - float(near.mean())

    floor_vals, floor_ok = [], []
    for u, v, up in pick:
        vt = truth_l.get(up, -1)
        if vt < 0:
            continue
        floor_vals.append(vt not in adjR.get(v, ()) and vt != v)
        floor_ok.append(vt)
    if not floor_vals:
        return resid_model, 0.0
    floor = float(np.mean(floor_vals))
    if floor >= 1.0:
        return resid_model, floor
    d = (resid_model - floor) / (1.0 - floor)
    return float(min(max(d, 0.0), 1.0)), floor
