# -*- coding: utf-8 -*-
"""NeuSymEA 重实现（NeurIPS 2025，Chen, Yuan, Zhang, Hua, Cao, Huang）。

官方仓库 `chensyCN/NeuSymEA-NeurIPS25`（GPL-3.0）。原栈为
`tensorflow==2.7.0` / `keras==2.7.0`（**仅支持 Python ≤ 3.9**），本机与云端
（Py3.12，无 conda）均无法直接安装 ⇒ 按本文 B 轨既有路线**在契约内重实现**。
官方源码已逐字取回并落盘 `.workbuddy/neusymea_src/`，本文件逐条对照实现。

**方法结构（`run-neusym.py::align` ＋ `objects/KGs.py`）**
神经-符号**迭代互注**：每轮「神经 EA 模型训练 → 其互近邻预测注入为高置信锚点
（概率 $\\delta_0=0.5$）→ 符号概率推理迭代 $10$ 轮」；循环 $5$ 轮后收官。

**忠实实现的构件**
1. **神经侧＝Dual-AMN**：官方 `config.py` 的 `ea_model` 默认值即 `"dualamn"`，
   且 `ea/` 目录自带 `dualamn_layers.py`。本实现直接继承本文既有
   `code/baselines/recent/dual_amn.py::DualAMN`（编码器 + NHSM 损失 + 隐式对齐），
   以落实"神经侧无须从零重写"。
2. **锚点集合的取法**（官方 `KGsUtil.generate_input_for_emb_model`）：
   训练集 = 「当前仍为 argmax 匹配」的 `annotated_alignments`，即
   `{(l, r) ∈ annotated : sub_ent_match[l] == r}`。
3. **神经预测＝未对齐实体间的互近邻**（官方 `DualAmn.train` 末段）：
   对两侧未对齐实体集做互相最近邻，取互为最优者为新伪对。
4. **注入规则**（官方 `KGs.inject_ea_inferred_pairs`）：伪对以固定的
   $\\delta_0$（`config.py` 默认 $0.5$）**无条件覆盖**当前匹配，但**已在
   `annotated_alignments` 中的对（即种子与前轮已注入者）不被覆盖**。
5. **符号概率推理**（官方 `probabilisticReasoning.py` ＋ `KGs.__run_per_iteration`）：
   - `register_ent_equality`：由关系的对齐概率与**功能度**（relation
     functionality $=|\\mathrm{head\\_set}|/\\mathrm{freq}$）给出似然因子；
   - `register_ongoing_prob_product` / `register_rel_align_prob_norm`：累积关系
     对齐证据；`__update_rel_align_dict` 以 $\\mathrm{score}/(\\mathrm{const}+\\mathrm{norm})$
     转为关系对齐概率（$\\mathrm{const}=10$）；
   - `update_ent_align_prob`：以 $1-p$ 为分取 argmax 作为该实体的匹配；
   - `__ent_bipartite_matching`：强制 1-1（互选最优，非互选者清空）。
   - 超参完全照抄：$\\theta=0.1$、$\\epsilon=1.01$、$\\delta=0.01$、$\\mathrm{const}=10$。
   - `init = (_iter_num <= 1)`（对应官方 `not has_load and _iter_num <= 1`）。
6. **末段收官**：循环后再跑一轮符号推理（官方 `kgs.run()`），并在末态锚点上
   再做一次神经训练（官方 `fine_tune()` 的对应物）。

**简化项与替换项（如实披露，逐条进 `FIDELITY.md`）**
- **单进程、快照语义**：官方用 `multiprocessing` 分片，各 worker 在**自身副本**上
  就地更新，故"同一轮内先更新的实体会影响同 worker 后面读取的实体"，其结果
  随 worker 划分而变、原实现本身不可复现。本实现统一改为**按迭代起始快照计算、
  迭代末合并**，从而确定可复现。实体遍历顺序用 `seed` 固定的 RNG（官方用全局
  `random.shuffle`）。
- **候选取自"全部未对齐实体"**：官方 `rest_set` 由 `dev_pair`（＝其内部测试集）
  决定；本文契约禁止读 `pair.test`，故改为「全体实体 − 已锚定实体」。不使用任何
  测试标签。
- **CSLS → 余弦**：官方预测用 CSLS 缓解 hubness；本实现按本文统一打分口径改用
  余弦（与其余方法逐字符同式），与 `dual_amn.py` 的同名替换一致。
- **无字面量节点**：官方 KG 同时收纳属性三元组并产生 literal 节点
  （`is_literal_list`）；DBP15K 官方版 `attr_triples_{1,2}` 为空文件（实测 0 B），
  故本实现无 literal 分支。
- **终局打分口径**：官方框架的最终输出是**符号映射**（`sub_ent_match/sup_ent_match`
  ＋概率），其自评是"阈值下的 precision/recall"。本文统一口径要求**全候选池排名**，
  故取「符号确认的配对置为 $1+p$（恒 $>1\\ge$ 余弦上界，故优先），其余候选按神经
  嵌入余弦排序」—— 这使 Hits@1 恰为符号映射的命中率，同时 Hits@K/MRR 仍有稠密分辨。
  此融合规则为**本文口径统一所加**，非官方原式。
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cussm.model import _rownorm
from cussm.torch_backend import get_device

from ._gnn import Graph, mine_mutual
from .dual_amn import DualAMN


# ================================================================ 符号层（PRASE 移植）
class _SymbolicState:
    """`probabilisticReasoning.py` ＋ `KGs` 概率推理部分的移植（单进程）。

    关系 id 约定与 `_gnn.Graph` 一致：正向边用 `r`、反向边用 `r + n_rel`；
    右侧图的关系 id 再整体平移 `2*n_rel_max`，使左右关系空间不相交
    （对应官方 `generate_input_for_emb_model` 里的 `rel_bias`）。
    """

    def __init__(self, gL: Graph, gR: Graph, n_rel_max: int, theta: float = 0.1,
                 epsilon: float = 1.01, delta: float = 0.01, const: float = 10.0,
                 seed: int = 2026):
        self.gL, self.gR = gL, gR
        self.rel_bias = 2 * int(n_rel_max)
        self.theta, self.epsilon = float(theta), float(epsilon)
        self.delta, self.const = float(delta), float(const)
        self.rng = np.random.RandomState(int(seed) % (2 ** 31 - 1))
        self._iter_num = 0

        self.sub_match: list = [None] * int(gL.n)      # 左实体 → 右实体
        self.sub_prob: list = [0.0] * int(gL.n)
        self.sup_match: list = [None] * int(gR.n)      # 右实体 → 左实体
        self.sup_prob: list = [0.0] * int(gR.n)

        self.rel_align_l: dict = {}
        self.rel_align_r: dict = {}
        self._build_facts()

    # ---- 结构：由边索引一次算好（对应 KG.init 里的 fact_dict / functionality） ----
    def _build_facts(self):
        def build(g: Graph, bias: int):
            by_head, by_tail = {}, {}
            hset, tset, cnt = {}, {}, {}
            for s, d, r in zip(g.src, g.dst, g.rel):
                rid = int(r) + bias
                si, di = int(s), int(d)
                by_head.setdefault(si, []).append((rid, di))
                by_tail.setdefault(di, []).append((rid, si))
                hset.setdefault(rid, set()).add(si)
                tset.setdefault(rid, set()).add(di)
                cnt[rid] = cnt.get(rid, 0) + 1
            func = {k: len(hset[k]) / cnt[k] for k in cnt}
            f_inv = {k: len(tset[k]) / cnt[k] for k in cnt}
            return by_head, by_tail, func, f_inv

        self.l_head, self.l_tail, self.l_func, self.l_func_inv = build(self.gL, 0)
        self.r_head, self.r_tail, self.r_func, self.r_func_inv = build(self.gR, self.rel_bias)

    # ---- 工具（probabilisticReasoning 顶部的四个小函数） ----
    @staticmethod
    def _pair(match, prob, idx):
        c = match[idx]
        return (None, 0.0) if c is None else (c, prob[idx])

    def _rel_align_prob(self, d: dict, a, b) -> float:
        sub = d.get(a)
        if not sub:
            return 0.0
        p = sub.get(b, 0.0)
        return 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)

    def _register_ent_equality(self, ongoing: dict, ra_l: dict, ra_r: dict,
                               func_l: dict, func_r: dict,
                               rel, rel_c, tail_c, head_p: float, init: bool) -> None:
        """`register_ent_equality` 的逐行移植。"""
        eps = self.epsilon
        prob_sub = self._rel_align_prob(ra_l, rel, rel_c) / eps
        prob_sup = self._rel_align_prob(ra_r, rel_c, rel) / eps
        if prob_sub < self.theta and prob_sup < self.theta:
            if init:
                prob_sub = prob_sup = self.theta      # 首两轮允许"冷启动"
            else:
                return
        fl = func_l.get(rel, 0.0) / eps
        fr = func_r.get(rel_c, 0.0) / eps
        factor = 1.0
        f_l = 1.0 - head_p * prob_sup * fr
        f_r = 1.0 - head_p * prob_sub * fl
        if prob_sub >= 0.0 and fl >= 0.0:
            factor *= f_l
        if prob_sup >= 0.0 and fr >= 0.0:
            factor *= f_r
        if 1.0 - factor > self.delta:
            ongoing[tail_c] = ongoing.get(tail_c, 1.0) * factor

    @staticmethod
    def _update_ent_align_prob(ongoing: dict) -> tuple:
        """`update_ent_align_prob` 的逐行移植（`fusion_func=None` ⇒ 不融合嵌入）。"""
        val, cand = 0.0, None
        for c, p in ongoing.items():
            v = 1.0 - p
            if v >= val:                              # 与官方同：取「最后出现的最大」
                val, cand = v, c
        val = 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)
        return cand, val

    # ---- 单方向一轮（one_iteration_one_way 的批处理语义版） ----
    def _one_way(self, side: str, init: bool, ent_align: bool = True) -> None:
        if side == "L":
            g, o_head = self.gL, self.r_head
            my_tail, my_func = self.l_tail, self.l_func
            o_func = self.r_func
            match, prob = self.sub_match, self.sub_prob
            ra_l, ra_r = self.rel_align_l, self.rel_align_r
            n_other = int(self.gR.n)
        else:
            g, o_head = self.gR, self.l_head
            my_tail, my_func = self.r_tail, self.r_func
            o_func = self.l_func
            match, prob = self.sup_match, self.sup_prob
            ra_l, ra_r = self.rel_align_r, self.rel_align_l
            n_other = int(self.gL.n)

        is_lit = [False] * n_other                     # DBP15K 无 literal 节点
        ents = list(range(int(g.n)))
        self.rng.shuffle(ents)

        snap_m, snap_p = list(match), list(prob)       # 迭代起始快照（见模块 docstring）
        rel_ongoing: dict = {}
        rel_norm: dict = {}
        upd: dict = {}

        for e in ents:
            ongoing: dict = {}
            for (rel, head) in my_tail.get(e, ()):
                hc, hp = self._pair(snap_m, snap_p, head)
                if hc is None or hp < self.theta:
                    continue
                ec, tp = self._pair(snap_m, snap_p, e)
                if ec is not None:
                    rel_norm[rel] = rel_norm.get(rel, 0.0) + hp * tp
                for (rel_c, tail_c) in o_head.get(hc, ()):
                    if is_lit[tail_c]:
                        continue
                    eqv = tp if tail_c == ec else 0.0
                    if eqv > 0.0:
                        d = rel_ongoing.setdefault(rel, {})
                        d[rel_c] = d.get(rel_c, 0.0) + hp * eqv
                    if ent_align:
                        self._register_ent_equality(ongoing, ra_l, ra_r, my_func, o_func,
                                                   rel, rel_c, tail_c, hp, init)
            if ent_align:
                cand, val = self._update_ent_align_prob(ongoing)
                if cand is not None and val >= snap_p[e]:
                    upd[e] = (cand, val)

        # ---- 迭代末合并（对应 __update_rel_align_dict 与 __merge_ent_align_result） ----
        ra_l.clear()
        for rel, d in rel_ongoing.items():
            norm = rel_norm.get(rel, 1.0)
            ra_l[rel] = {rc: sc / (self.const + norm) for rc, sc in d.items()}
        for e, (c, v) in upd.items():
            match[e], prob[e] = c, v

    def _bipartite(self) -> None:
        """`__ent_bipartite_matching` 的逐行移植：强制 1-1。"""
        for l in range(len(self.sub_match)):
            c = self.sub_match[l]
            if c is None:
                continue
            if self.sup_prob[c] < self.sub_prob[l]:
                self.sup_match[c], self.sup_prob[c] = l, self.sub_prob[l]
        for l in range(len(self.sub_match)):
            c = self.sub_match[l]
            if c is None:
                continue
            if self.sup_match[c] is None:
                continue
            if self.sup_match[c] != l:
                self.sub_match[l], self.sub_prob[l] = None, 0.0

    # ---- 对外 ----
    def run(self, iters: int, deadline: float | None = None, on_step=None) -> int:
        """对应 `KGs.run()`：每轮「左向 → 1-1 匹配 → 右向」，共 `iters` 轮。"""
        done = 0
        for i in range(int(iters)):
            if deadline and time.time() > deadline:
                break
            self._iter_num = i
            init = self._iter_num <= 1                 # has_load 恒 False
            self._one_way("L", init=init)
            self._bipartite()
            self._one_way("R", init=init, ent_align=False)
            done += 1
            if on_step:
                on_step(done)
        return done

    def inject(self, pairs, prob: float, annotated: set) -> int:
        """`inject_ea_inferred_pairs(pairs, bias, filter=False, reinject=True)` 的移植。"""
        n = 0
        p = float(prob)
        for (l, r) in pairs:
            if (l, r) in annotated:                    # reinject=True：不动已注入者
                continue
            self.sub_match[l], self.sub_prob[l] = r, p
            self.sup_match[r], self.sup_prob[r] = l, p
            annotated.add((l, r))
            n += 1
        return n


# ================================================================ 契约内方法
class NeuSymEA(DualAMN):
    """NeuSymEA（神经-符号迭代互注）—— 见模块 docstring 的忠实/简化清单。"""

    name = "NeuSymEA（NeurIPS 2025，重实现）"
    impl = "reimplemented-from-paper"
    uses_seeds_for_align = True         # 神经侧有监督；符号侧由种子初始化
    uses_labse = False                  # 神经侧＝Dual-AMN basic：随机初始化嵌入
    uses_momentum = False
    index_input = True
    align_mode = "identity"

    def __init__(self, n_rounds: int = 5, sym_iters: int = 10,
                 theta: float = 0.1, epsilon: float = 1.01, delta: float = 0.01,
                 const: float = 10.0, inj_prob: float = 0.5, **kw):
        super().__init__(**kw)
        self.n_rounds = max(int(n_rounds), 1)
        self.sym_iters = max(int(sym_iters), 0)
        self.theta, self.epsilon = float(theta), float(epsilon)
        self.delta, self.const = float(delta), float(const)
        self.inj_prob = float(inj_prob)
        self._anchors = np.zeros((0, 2), dtype=np.int64)
        self.sym_map: dict = {}
        self.sym_prob: list = []
        self.round_log: list = []
        self.n_sym_iters = 0
        self.n_injected = 0

    # ---------------------------------------------------------------- 锚点与预测
    def _current_anchors(self, sym: _SymbolicState, annotated: set) -> np.ndarray:
        """`generate_input_for_emb_model`：训练集 = 仍为 argmax 匹配的 annotated 对。"""
        A = [(l, r) for (l, r) in annotated if sym.sub_match[l] == r]
        if not A:
            return np.zeros((0, 2), dtype=np.int64)
        return np.asarray(sorted(A), dtype=np.int64).reshape(-1, 2)

    def _train_anchors(self, opt, xL, xR, tL, tR, nL, nR, dev, n_ep: int) -> int:
        A = self._anchors
        S = int(len(A))
        if S < 4:
            return 0
        B = max(4, min(int(self.batch), S))
        steps = 0
        for _ in range(int(n_ep)):
            perm = torch.randperm(S, device=dev)
            for i in range(0, S, B):
                idx = perm[i:i + B]
                if int(idx.numel()) < 4:
                    continue
                s = torch.as_tensor(A[idx.cpu().numpy()], device=dev)
                ZL = self._encode(xL, tL, nL, "L")
                ZR = self._encode(xR, tR, nR, "R")
                zl = F.normalize(ZL[s[:, 0]], dim=1)
                zr = F.normalize(ZR[s[:, 1]], dim=1)
                loss = self._nhsm(zl, zr) + self._nhsm(zr, zl)   # Dual-AMN 的 NHSM
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.enc.parameters(), 5.0)
                opt.step()
                steps += 1
        self.steps_done += steps
        self.epochs_done += int(n_ep)
        return steps

    def _predict_pairs(self, xL, xR, tL, tR, nL, nR, dev, annotated: set):
        """`DualAmn.train` 末段：未对齐实体间的互近邻（CSLS → 本文统一余弦）。"""
        with torch.no_grad():
            ZL = self._encode(xL, tL, nL, "L").float()
            ZR = self._encode(xR, tR, nR, "R").float()
        mL = np.ones(nL, dtype=bool)
        mR = np.ones(nR, dtype=bool)
        for (l, r) in annotated:
            mL[l] = False
            mR[r] = False
        cl = np.nonzero(mL)[0]
        cr = np.nonzero(mR)[0]
        if len(cl) < 2 or len(cr) < 2:
            return [], 0
        dL = torch.as_tensor(cl, device=dev)
        dR = torch.as_tensor(cr, device=dev)
        iL, iR = mine_mutual(ZL[dL], ZR[dR])
        iL, iR = iL.cpu().numpy(), iR.cpu().numpy()
        return ([(int(cl[a]), int(cr[b])) for a, b in zip(iL, iR)], int(len(cl)))

    # ---------------------------------------------------------------- 拟合
    def fit(self, pair):
        t0 = time.time()
        dev = get_device(self.device)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        gL, gR = Graph(pair.left), Graph(pair.right)
        nL, nR = int(pair.left.n), int(pair.right.n)
        xL = torch.arange(nL, device=dev)              # index_input=True ⇒ 只给下标
        xR = torch.arange(nR, device=dev)
        self.feat_source = {"left": self.input_tag, "right": self.input_tag}
        tL, tR = gL.tensors(dev), gR.tensors(dev)
        n_rel = 2 * max(gL.n_rel, gR.n_rel)
        self.enc = self._make_encoder(None, n_rel, nL, nR, dev)
        opt = torch.optim.Adam(self.enc.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        deadline = (time.time() + float(self.max_minutes) * 60.0
                    if self.max_minutes else None)

        # ---- 符号层初始化：种子以概率 1.0 注入（对应 KGs.__init） ----
        sym = _SymbolicState(gL, gR, max(gL.n_rel, gR.n_rel), self.theta,
                             self.epsilon, self.delta, self.const, seed=self.seed)
        annotated: set = set()
        seeds = np.asarray(pair.seeds, dtype=np.int64).reshape(-1, 2)
        for l, r in seeds:
            li, ri = int(l), int(r)
            sym.sub_match[li], sym.sub_prob[li] = ri, 1.0
            sym.sup_match[ri], sym.sup_prob[ri] = li, 1.0
            annotated.add((li, ri))

        total_ep = max(int(self.epochs), 1)
        ep_r = max(1, total_ep // self.n_rounds)

        # ---- 迭代互注：神经训练 → 互近邻预测 → 注入 → 符号推理 ----
        self._anchors = self._current_anchors(sym, annotated)
        for rnd in range(self.n_rounds):
            if deadline and time.time() > deadline:
                break
            _t = time.time()
            st = self._train_anchors(opt, xL, xR, tL, tR, nL, nR, dev, ep_r)
            t_net = time.time() - _t
            _t = time.time()
            pairs, n_cand = self._predict_pairs(xL, xR, tL, tR, nL, nR, dev, annotated)
            t_pred = time.time() - _t
            _t = time.time()
            inj = sym.inject(pairs, self.inj_prob, annotated)
            self.n_injected += inj
            ns = sym.run(self.sym_iters, deadline=deadline)
            t_sym = time.time() - _t
            self.n_sym_iters += ns
            self._anchors = self._current_anchors(sym, annotated)
            self.round_log.append({
                "轮": rnd + 1, "神经轮数": int(ep_r), "梯度步数": int(st),
                "候选实体": int(n_cand), "神经预测对数": int(len(pairs)),
                "注入对数": int(inj), "锚点数": int(len(self._anchors)),
                "符号推理轮数": int(ns),
                "秒_神经": round(t_net, 1), "秒_预测": round(t_pred, 1),
                "秒_符号": round(t_sym, 1)})
            print(f"    [NeuSymEA] 轮 {rnd + 1}/{self.n_rounds}：步 {st}、"
                  f"候选 {n_cand}、预测 {len(pairs)}、注入 {inj}、"
                  f"锚点 {len(self._anchors)}；"
                  f"秒(神经 {t_net:.1f} / 预测 {t_pred:.1f} / 符号 {t_sym:.1f})",
                  flush=True)

        # ---- 收官：末轮符号推理 + 末态锚点上再训（官方 kgs.run() + fine_tune()） ----
        self.n_sym_iters += sym.run(self.sym_iters, deadline=deadline)
        self._anchors = self._current_anchors(sym, annotated)
        self._train_anchors(opt, xL, xR, tL, tR, nL, nR, dev, ep_r)
        self.train_time = time.time() - t0

        with torch.no_grad():
            ZLt = self._encode(xL, tL, nL, "L").float()
            ZRt = self._encode(xR, tR, nR, "R").float()
            W = self._align(pair, ZLt, ZRt)            # identity ⇒ 单位阵
        self.W = W
        self.ZLs = _rownorm(ZLt.cpu().numpy() @ W)
        self.ZRs = _rownorm(ZRt.cpu().numpy())
        self.sym_map = {l: sym.sub_match[l] for l in range(nL)
                        if sym.sub_match[l] is not None}
        self.sym_prob = list(sym.sub_prob)
        return self

    # ---------------------------------------------------------------- 输出契约
    def score_block(self, pair, left_idx):
        """神经余弦为底；符号确认的配对置 $1+p$（恒 $>1$，故恒优于任何余弦）。"""
        S = (self.ZLs[left_idx] @ self.ZRs.T).astype(np.float32)
        if self.sym_map:
            for i, l in enumerate(np.asarray(left_idx).ravel()):
                r = self.sym_map.get(int(l))
                if r is not None:
                    S[i, int(r)] = np.float32(1.0 + max(float(self.sym_prob[int(l)]), 1e-6))
        return S

    def flags(self) -> dict:
        f = super().flags()
        f["迭代互注轮数"] = int(self.n_rounds)
        f["符号推理轮数（合计）"] = int(self.n_sym_iters)
        f["神经训练轮数（合计）"] = int(self.epochs_done)
        f["注入伪对数（合计）"] = int(self.n_injected)
        f["末态锚点数"] = int(len(self._anchors))
        f["末态配对数"] = int(len(self.sym_map))
        f["符号超参"] = (f"θ={self.theta:g}, ε={self.epsilon:g}, "
                        f"δ={self.delta:g}, const={self.const:g}, δ0={self.inj_prob:g}")
        f["预算分配"] = (f"总神经轮数 {int(self.epochs)} ÷ {int(self.n_rounds)} 轮 "
                        f"= 每轮 {max(1, int(self.epochs) // int(self.n_rounds))} 轮；"
                        f"每轮符号推理 {int(self.sym_iters)} 轮")
        if self.round_log:
            f["分阶段秒数（神经｜预测｜符号）"] = "；".join(
                f"轮{r['轮']} {r['秒_神经']}｜{r['秒_预测']}｜{r['秒_符号']}"
                for r in self.round_log)
            f["各轮锚点轨迹"] = "→".join(str(r["锚点数"]) for r in self.round_log)
        return f


__all__ = ["NeuSymEA"]
