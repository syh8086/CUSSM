# -*- coding: utf-8 -*-
"""基线方法 —— 与本文方法共用同一份输入特征与同一套输出契约。

纳入准则（对应第 5.2 节）：① 有公开实现或论文可复现；② 能吃到同一份 `Pair` 输入；
③ 覆盖"表层字面 / 属性语义 / 纯结构嵌入 / 跨语言联合嵌入 / 联合+自训练"五个梯队，
避免只与弱基线比较。

| 基线 | 类别 | 真训练？ | 参考 |
|---|---|---|---|
| NNSim        | 表层字面（无结构） | 否 | 经典 sanity check |
| AttrSim      | 属性语义（无结构） | 否 | 属性对齐的朴素上限 |
| TransE-NN    | 两侧独立训练 + 事后 Procrustes | 是 | Bordes et al., NIPS 2013 |
| MTransE      | 结构嵌入 + 跨语言**联合**训练 | 是 | Chen et al., IJCAI 2017 |
| JAPE         | 结构 + 属性联合嵌入 | 是 | Sun et al., WWW 2017 |
| BootEA-lite  | MTransE + 迭代自训练（邻居一致性编辑） | 是 | Sun et al., IJCAI 2018 |

**公平性**：所有"可训练"基线都走**同一个训练器** `torch_backend.train_joint`
（同样的轮数、负采样数、学习率与设备），差别只在"吃什么视图、怎么融合"；
不存在的"给本文方法喂更好特征"或"给基线减轮数"的情况。
超参（融合权重、迭代轮数）一律在**同一个验证种子集**上选，绝不使用测试集。

**关于 TransE-NN 的说明（重要）**：两侧**各自独立**训练 TransE、再用种子做正交
Procrustes，是本项目实测过的"朴素迁移"路线，在 DBP15K fr_en 上全候选集 Hits@1
只有 0.005（随机水平约 1e-5）。这不是实现 bug —— 独立训练的 TransE 在自身三元组
上的 filtered link-prediction MRR 达 0.15–0.29。根因是 TransE 的目标 ‖h+r−t‖→0
**不正交不变**，两个独立收敛的嵌入空间之间不存在可用的正交映射。保留该基线并
如实报告其失败，正是"必须联合训练"这一设计动机的直接证据。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from cussm.data import Pair
from cussm.features import _hash_bow, _l2_sp, _tfidf, build_view
from cussm.model import BaseMatcher, _rownorm, ortho_procrustes
from cussm.propagation import StructProp
from cussm.torch_backend import get_device, has_torch, train_joint


# ================================================================ 非训练基线
def _sp_scores(ZL_c, ZR) -> np.ndarray:
    """稀疏余弦得分块 (m, N)。稀疏乘积必须先 `.toarray()` ——
    `np.asarray(sparse, dtype=...)` 会抛 "setting an array element with a sequence"。"""
    return (ZR @ ZL_c.T).T.toarray().astype(np.float32)


class NNSim(BaseMatcher):
    name = "NNSim（表层名）"
    family = "表层字面"

    def __init__(self, dim: int = 1024, seed: int = 2026):
        self.dim, self.seed = dim, seed

    def fit(self, pair: Pair):
        # 两侧共用同一散列宽度与散列函数 → 特征直接可比（约束 R1）
        self.ZL = _l2_sp(_tfidf(_hash_bow(pair.left.surfaces(), self.dim,
                                          n_items=pair.left.n)))
        self.ZR = _l2_sp(_tfidf(_hash_bow(pair.right.surfaces(), self.dim,
                                          n_items=pair.right.n)))
        return self

    def score_block(self, pair, left_idx):
        return _sp_scores(self.ZL[left_idx], self.ZR)


class AttrSim(BaseMatcher):
    name = "AttrSim（属性语义）"
    family = "属性语义"

    def __init__(self, dim: int = 1024, seed: int = 2026):
        self.dim, self.seed = dim, seed

    def fit(self, pair: Pair):
        def vals(kg):
            s = [[] for _ in range(kg.n)]
            for e, _k, v, _lg in kg.att_triples:
                if 0 <= e < kg.n:
                    s[e].append(v)
            return [" ; ".join(x) for x in s]
        self.ZL = _l2_sp(_tfidf(_hash_bow(vals(pair.left), self.dim,
                                          n_items=pair.left.n)))
        self.ZR = _l2_sp(_tfidf(_hash_bow(vals(pair.right), self.dim,
                                          n_items=pair.right.n)))
        return self

    def score_block(self, pair, left_idx):
        return _sp_scores(self.ZL[left_idx], self.ZR)


# ================================================================ 结构传播族
class _StructBase(BaseMatcher):
    """以"种子锚定的跨图结构画像"为唯一得分的基线族公共实现。"""
    family = "结构传播"

    def __init__(self, rounds: int = 0, rounds_grid=None, conf_q: float = 55.0,
                 max_new: int = 3000, chunk: int = 512, seed: int = 2026, **kw):
        self.rounds, self.rounds_grid = rounds, rounds_grid
        self.conf_q, self.max_new, self.chunk, self.seed = conf_q, max_new, chunk, seed
        self.train_time = 0.0
        self.SP = None
        self.rounds_selected = rounds

    def _fit_sp(self, pair: Pair):
        import time
        t0 = time.time()
        grid = (self.rounds,) if not self.rounds_grid else tuple(self.rounds_grid)
        best, best_h = None, -1.0
        if len(grid) > 1:
            vi, vj = pair.val[:, 0], pair.val[:, 1]
            for r in grid:
                sp = StructProp(rounds=r, conf_q=self.conf_q, max_new=self.max_new,
                                chunk=self.chunk, seed=self.seed).fit(pair)
                h = 0
                for i in range(0, len(vi), 512):
                    S = sp.score_block(vi[i:i + 512])
                    h += int((np.argmax(S, axis=1) == vj[i:i + 512]).sum())
                    del S
                h /= max(len(vi), 1)
                if h > best_h + 1e-12:
                    best, best_h, self.rounds_selected = sp, h, r
        else:
            self.SP = StructProp(rounds=grid[0], conf_q=self.conf_q,
                                 max_new=self.max_new, chunk=self.chunk,
                                 seed=self.seed).fit(pair)
            self.rounds_selected = grid[0]
            self.train_time = time.time() - t0
            return self
        self.SP = best
        self.train_time = time.time() - t0
        return self

    def score_block(self, pair, left_idx):
        return self.SP.score_block(left_idx)

    def param_detail(self) -> dict:
        return {"结构传播新增匹配对数": int(0 if self.SP is None
                                    else self.SP.nM - self.SP.n_seed)}

    def n_params(self) -> int:
        return 0

    def flags(self) -> dict:
        return {"传播轮数(验证集选出)": int(self.rounds_selected),
                "新增匹配对数": int(0 if self.SP is None
                                 else self.SP.nM - self.SP.n_seed)}


class StructSim(_StructBase):
    """结构画像，**不做迭代**：只用训练种子做锚。"""
    name = "StructSim（结构画像，无迭代）"

    def fit(self, pair: Pair):
        self.rounds_grid = None
        self.rounds = 0
        return self._fit_sp(pair)


class StructPropK(_StructBase):
    """迭代结构传播（自训练）：每轮把互近邻高置信对并入锚集。"""
    name = "StructProp（迭代自训练）"

    def fit(self, pair: Pair):
        self.rounds_grid = (1, 3, 5)
        return self._fit_sp(pair)


class LinFuse(BaseMatcher):
    """两视图**线性融合**：结构画像 + 属性/字面视图，融合权重在验证集上选。

    这是"两条路由但不要纤维/类型/粘合、也不做超参联合选择"的对照 ——
    用来隔离 CUSSM 里"纤维硬约束 + 类型先验 + 粘合项 + 联合调参"各自的贡献。
    """
    name = "LinFuse（两视图线性融合）"
    family = "线性融合"

    def __init__(self, d_sem: int = 256, seed: int = 2026, chunk: int = 512, **kw):
        self.d_sem, self.seed, self.chunk = d_sem, seed, chunk
        self.train_time = 0.0
        self.alpha = 0.5

    def fit(self, pair: Pair):
        import time
        from cussm.features import build_view
        from cussm.model import _zscore, ortho_procrustes, ALPHA_GRID
        from cussm import mem
        t0 = time.time()
        self.SP = StructProp(rounds=0, chunk=self.chunk, seed=self.seed).fit(pair)
        sl, sr = pair.seeds[:, 0], pair.seeds[:, 1]
        self.VL = build_view(pair.left, 64, self.d_sem, self.seed)
        self.VR = build_view(pair.right, 64, self.d_sem, self.seed)
        # 与 CUSSM 用**同一套**语义路由定义（两块分别对齐），差别只在没有纤维/类型/粘合
        # 与联合调参 —— 这样 LinFuse 才是干净的"去掉三个附加通道"的对照。
        Wn = ortho_procrustes(self.VL.sem_name[sl], self.VR.sem_name[sr])
        Wv = ortho_procrustes(self.VL.sem_val[sl], self.VR.sem_val[sr])
        self.NL, self.NR = _rownorm(self.VL.sem_name @ Wn), self.VR.sem_name
        self.AL_, self.AR_ = _rownorm(self.VL.sem_val @ Wv), self.VR.sem_val
        vi, vj = pair.val[:, 0], pair.val[:, 1]
        # ---- 按行分块（2026-09-29 消除 OOM）----
        # 原实现对**全量 val**物化 (n_val, N_R) 的 AN/AV，并在其上跑 w 网格与
        # ALPHA_GRID 网格：Track A′（N_R≈1.9e5、n_val≈5.4e3）上算术和达 119.9 GB，
        # 实测被内核 OOM 杀死；yago3_10 亦达 8.3 GB。改为**外块内网格**——
        # 每块算齐读数后跑完整张网格、只累加整数命中数 ⇒ 峰值与 n_val 解耦。
        # 逐行运算（z-score / argmax）按行独立，故读数逐位不变（见 cussm/mem.py）。
        n_val = int(len(vi))
        CH = mem.plan_block(self.VR.n, arrays=4, chunk=self.chunk)

        # 阶段 1：语义路由内部配比 w（只依赖两视图余弦，与 zH/zK 无关）
        W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
        acc_w = [0] * len(W_GRID)
        for sl in mem.block_slices(n_val, CH):
            A = (self.NL[vi[sl]] @ self.NR.T).astype(np.float32)
            B = (self.AL_[vi[sl]] @ self.AR_.T).astype(np.float32)
            vb = vj[sl]
            for i, w in enumerate(W_GRID):
                acc_w[i] += int((np.argmax(w * A + (1 - w) * B, axis=1) == vb).sum())
            del A, B
        bw, bh = 0.5, -1.0
        for i, w in enumerate(W_GRID):
            h = acc_w[i] / max(n_val, 1)
            if h > bh + 1e-12:
                bw, bh = float(w), h
        self.w = bw

        # 阶段 2：融合权重 α（zH/zK 都是**逐行**标准化 ⇒ 分块与整块逐位相同）
        acc_a = [0] * len(ALPHA_GRID)
        for sl in mem.block_slices(n_val, CH):
            zb = vi[sl]
            zH = _zscore(self.SP.score_block(zb))
            A = (self.NL[zb] @ self.NR.T).astype(np.float32)
            B = (self.AL_[zb] @ self.AR_.T).astype(np.float32)
            zK = _zscore(bw * A + (1.0 - bw) * B)
            vb = vj[sl]
            # 与 CUSSM 共用同一张网格（见 cussm.model.ALPHA_GRID）
            for i, a in enumerate(ALPHA_GRID):
                acc_a[i] += int((np.argmax(a * zH + (1.0 - a) * zK, axis=1) == vb).sum())
            del zH, zK, A, B
        best = (-1.0, 0.5)
        for i, a in enumerate(ALPHA_GRID):
            h = acc_a[i] / max(n_val, 1)
            if h > best[0] + 1e-12:
                best = (h, float(a))
        self.alpha, self.val_hits1 = best[1], best[0]
        self.train_time = time.time() - t0
        return self

    def _route_H(self, left_idx):
        return self.SP.score_block(left_idx)

    def _route_K(self, left_idx):
        return (np.float32(self.w) * (self.NL[left_idx] @ self.NR.T)
                + np.float32(1.0 - self.w) * (self.AL_[left_idx] @ self.AR_.T)
                ).astype(np.float32)

    def routes(self, pair, left_idx):
        """暴露两条路由的**原始读数**（未标准化），供 SPS 的 Δ_nat / Δ_glue 审计使用。"""
        return {"H 结构路由": self._route_H(left_idx),
                "K 语义路由": self._route_K(left_idx)}

    def fusion_routes(self):
        """只有 α 与 1−α 都非零时两条路由才都进融合（与 CUSSM 用同一规则）。"""
        out = []
        if self.alpha > 0:
            out.append("H 结构路由")
        if self.alpha < 1:
            out.append("K 语义路由")
        return out

    def score_block(self, pair, left_idx):
        from cussm.model import _zscore
        H = self._route_H(left_idx)
        K = self._route_K(left_idx)
        return (self.alpha * _zscore(H) + (1.0 - self.alpha) * _zscore(K)).astype(np.float32)

    def param_detail(self) -> dict:
        return {"语义路由映射 ×2": int(2 * self.d_sem * self.d_sem), "融合标量 w, α": 2}

    def n_params(self) -> int:
        return int(sum(self.param_detail().values()))

    def flags(self) -> dict:
        return {"α(验证集选出)": round(float(self.alpha), 3),
                "w(验证集选出)": round(float(self.w), 3)}


# ================================================================ 结构嵌入族（负面对照）
class _TrainableBase(BaseMatcher):
    """共享同一训练器的基类。子类只决定"用什么视图、怎么融合"。"""
    family = "结构嵌入"
    uses_joint = True

    def __init__(self, dim: int = 128, epochs: int = 60, n_neg: int = 4,
                 lr: float = 1e-3, batch: int = 4096, align: str = "ortho",
                 lam_align: float = 1.0, d_sem: int = 256, seed: int = 2026,
                 device: str = "auto", max_minutes: float | None = None, **kw):
        self.dim, self.epochs, self.n_neg = dim, epochs, n_neg
        self.lr, self.batch, self.align, self.lam_align = lr, batch, align, lam_align
        self.d_sem, self.seed, self.device = d_sem, seed, device
        self.max_minutes = max_minutes
        self.train_time = 0.0
        self.epochs_done = None      # 实际完成的轮数（独立训练按两侧记；见 _fit_independent）
        self.emb = None
        self.eparams = 0

    # ---- 独立训练（TransE-NN 专用）----
    def _fit_independent(self, pair: Pair):
        import time
        import torch
        from cussm.torch_backend import _take, _transe_loss
        t0 = time.time()
        device = get_device(self.device)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        # **两侧独立训练必须与联合训练受同一个墙上时间上限约束**。
        # 2026-09-28 修正：原实现完全忽略 `max_minutes`，TransE-NN 固定跑满 `epochs`，
        # 于是三条结构嵌入基线处在**不同**预算下（破坏了"方法间只差方法本身"的比较
        # 前提），且结果随机器速度变化、**不可复现**（`--max-minutes` 的意义正是防这一点）。
        # 取"整法上限 ÷ 2"分给两侧：两侧规模相当，等分是自然选择。
        side_budget = None if not self.max_minutes else float(self.max_minutes) / 2.0
        outs, ep_done = [], []
        for kg in (pair.left, pair.right):
            n_e, n_r = kg.n, max(kg.n_rel, 1)
            T = np.asarray(kg.rel_triples, dtype=np.int64)
            E = (torch.randn(n_e, self.dim, device=device) * 0.1).requires_grad_(True)
            R = (torch.randn(n_r, self.dim, device=device) * 0.1).requires_grad_(True)
            tt = torch.as_tensor(T, device=device)
            opt = torch.optim.Adam([E, R], lr=self.lr)
            ts = time.time()
            n_done = 0
            if len(tt):
                for _ep in range(self.epochs):
                    if side_budget and (time.time() - ts) / 60 > side_budget:
                        break
                    p = rng.permutation(len(tt))
                    for i in range(0, len(tt), self.batch):
                        opt.zero_grad(set_to_none=True)
                        loss = _transe_loss(E, R, _take(tt, p, i, self.batch),
                                            n_e, self.n_neg, 1.0, 0.05)
                        loss.backward()
                        opt.step()
                        with torch.no_grad():
                            E.copy_(torch.nn.functional.normalize(E, dim=1))
                    n_done += 1
                    if side_budget and (time.time() - ts) / 60 > side_budget:
                        break
            outs.append(E.detach().cpu().numpy().astype(np.float32))
            ep_done.append(n_done)
            print(f"    [独立训练] {kg.name}: 完成 {n_done}/{self.epochs} 轮，"
                  f"用时 {time.time() - ts:.0f}s", flush=True)
        EL, ER = outs
        self.EL, self.ER = _rownorm(EL), _rownorm(ER)
        self.W = ortho_procrustes(self.EL[pair.seeds[:, 0]], self.ER[pair.seeds[:, 1]])
        self.ZLs = _rownorm(self.EL @ self.W)
        self.ZRs = self.ER
        self.eparams = (pair.left.n + pair.right.n) * self.dim * 4
        self.epochs_done = ep_done          # 供记录/复现核对实际完成的轮数
        self.train_time = time.time() - t0
        return self

    # ---- 联合训练（MTransE 家族）----
    def _fit_joint(self, pair: Pair):
        self.emb = train_joint(pair.left, pair.right, pair.seeds,
                               dim=self.dim, epochs=self.epochs, n_neg=self.n_neg,
                               lr=self.lr, batch=self.batch, align=self.align,
                               lam_align=self.lam_align, seed=self.seed,
                               device=get_device(self.device),
                               max_minutes=self.max_minutes)
        self.ZLs, self.ZRs = self.emb.views()
        self.train_time = self.emb.train_time
        self.eparams = (pair.left.n + pair.right.n) * self.dim * 4
        return self

    def score_block(self, pair, left_idx):
        return (self.ZLs[left_idx] @ self.ZRs.T).astype(np.float32)

    def n_params(self) -> int:
        return int(self.dim * self.dim) + 4

    def param_detail(self) -> dict:
        return {"对齐映射 W": int(self.dim * self.dim), "融合标量": 4,
                "结构嵌入表 E_L,E_R": int(self.eparams)}


class TransENN(_TrainableBase):
    """两侧独立训练 + 事后 Procrustes（朴素迁移基线，实测会失败，如实报告）。"""
    name = "TransE-NN（独立训练+Procrustes）"

    def fit(self, pair: Pair):
        return self._fit_independent(pair)


class MTransE(_TrainableBase):
    """结构嵌入 + 跨语言联合训练（正交对齐）。"""
    name = "MTransE（联合训练）"

    def fit(self, pair: Pair):
        return self._fit_joint(pair)


class JAPE(_TrainableBase):
    """结构联合嵌入 + 属性语义视图的联合对齐（JAPE 的主体部分）。

    原文以"结构嵌入 + 属性键嵌入"共享同一对齐映射；此处以
    `[结构联合嵌入 ‖ 属性/字面视图]` 的拼接空间做正交对齐来落实同一思想，
    省去原文的属性负采样细节（在 DBP15K 设定下收益甚微），第 5 章注明。
    """
    name = "JAPE（结构+属性联合）"
    family = "联合嵌入"

    def fit(self, pair: Pair):
        self._fit_joint(pair)
        VL = build_view(pair.left, 64, self.d_sem, self.seed)
        VR = build_view(pair.right, 64, self.d_sem, self.seed)
        self.JL = np.hstack([self.ZLs, VL.sem])
        self.JR = np.hstack([self.ZRs, VR.sem])
        sl, sr = pair.seeds[:, 0], pair.seeds[:, 1]
        self.Wj = ortho_procrustes(self.JL[sl], self.JR[sr])
        self.ZLs = _rownorm(self.JL @ self.Wj)
        self.ZRs = _rownorm(self.JR)
        return self

    def param_detail(self):
        d = super().param_detail()
        d["拼接空间对齐 W"] = int(self.JL.shape[1] ** 2)
        return d


BASELINES = [NNSim, AttrSim, StructSim, StructPropK, LinFuse,
             TransENN, MTransE, JAPE]
