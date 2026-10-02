# -*- coding: utf-8 -*-
"""SelfKG 重实现（WWW 2022，Liu et al., pp. 860–870, DOI 10.1145/3485447.3511945）。

**方法身份**：自监督实体对齐 —— **不使用任何种子对齐**（训练与对齐都不读 `pair.seeds`）。
原文的立论是：对齐的学习"从把未对齐的负例推远"中获益**多于**"把已对齐的正例拉近"，
据此给出**相对相似度度量（RSM）**，从而摆脱对正例标签的依赖。

**忠实实现的构件**（逐条对应 arXiv 2203.01044）
1. **RSM 目标**（式 2）：把正的相似度钉在其上界 $1/\\tau$，只对负例求和 —— 即
   $\\mathcal{L}=-\\log\\frac{e^{1/\\tau}}{e^{1/\\tau}+\\sum_i e^{s_i/\\tau}}$。本实现
   等价地写成"正 logit 固定为 $1/\\tau$、负 logit 为 $s_i/\\tau$ 的交叉熵"，
   与官方实现 `NCESoftmaxLoss`（label 恒为 0）同式。
2. **自负采样（self-negative sampling）**：负例**只从实体自己所在的图**里取。
   原文动机："跨图取负会把潜在对齐的实体当成负例（collision），最多掉 7.7%"。
3. **双负例队列**：两张图各维护一条 MoCo 式队列（官方 `neg_queue1` / `neg_queue2`），
   本实现每步把当前批的动量端嵌入入队，按"批"计长度。
4. **正例取动量端对同一批实体的表示**：官方 `pos_2 = self._model(pos_token)`，
   即"同一实体、另一编码器分支"。在 RSM 的视角下它近似于上界 $1/\\tau$。
5. **单层单头无关系 GAT**：原文"single-head graph attention network with **one layer**"，
   且消融发现"multi-hop information actually harms the performance"。故本实现的
   编码器用 `PlainGATEncoder`（不注入关系嵌入、层数钉 1）。
6. **LaBSE 统一空间**：原文用 LaBSE 把两图实体嵌入同一空间；本实现沿用
   `_gnn.name_features` 的 LaBSE 名嵌入（`flags()` 如实记录实际来源）。

**简化项（如实披露，正文表注须据此声明）**
- **描述文本缺失**：本文 loader 未载入 DBP15K 的 description 字段 ⇒ 只用了实体名一路。
- **队列长度**：原文队列 $1+K$ 批、负例可达 4k；本实现队列按批计（默认 6 批），
  负例规模见 `flags()['负例数(末次)']`。
- **动量 EMA 系数**：MoCo 系惯用 0.9999（对应 10 万级步数）。本文每次 `fit` 只有
  `epochs` 步（默认 60），故按步数定为 0.9（`_gnn._RecentBase` 的注记有推导）。
- **对齐步骤**：原文无显式对齐（uni-space 内直接比对）；本实现照此令 `W=I`
  （`align_mode="identity"`），不再在伪对上做 Procrustes。
- 原文"关系信息仅带来微小增益"，故本实现**不注入关系**，与原文主设定一致。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ._gnn import PlainGATEncoder, _RecentBase


class SelfKG(_RecentBase):
    """SelfKG（自监督）—— 见模块 docstring 的忠实/简化清单。"""

    name = "SelfKG（WWW 2022，重实现）"
    impl = "reimplemented-from-paper"
    uses_seeds_for_align = False        # 自监督：训练与对齐都**不碰 pair.seeds**
    uses_labse = True
    uses_momentum = True
    align_mode = "identity"             # 原文的 uni-space 隐式对齐

    def __init__(self, n_layer: int = 1, tau: float = 0.08, neg_batches: int = 6,
                 anchor_cap: int = 2048, **kw):
        # **层数钉 1**：原文明确单层，且报告多跳有害。调用方（如 `make_methods` 统一传
        # n_layer=2）传进来的值被有意忽略 —— 实际层数写进 `flags()['编码器层数']`。
        super().__init__(n_layer=1, tau=tau, **kw)
        self.n_layer_requested = int(n_layer)
        self.neg_batches = max(int(neg_batches), 1)
        self.anchor_cap = int(anchor_cap)
        # 两张图各一条负例队列（原文 neg_queue1 / neg_queue2）
        self._queue: dict = {"L": [], "R": []}
        self.step_log: list = []
        self.n_neg_last = 0

    # ---------------------------------------------------------------- 编码器
    def _make_encoder(self, d_in, n_rel, nL, nR, dev):
        return PlainGATEncoder(d_in, self.d, 1, self.drop).to(dev)

    # ---------------------------------------------------------------- 目标
    def _objective(self, ctx):
        dev, ep = ctx["dev"], int(ctx["ep"])
        # 左右**交替**取批：两条队列各自累积（与官方按 batch 交错遍历两个语言集等价）
        side = "L" if ep % 2 == 0 else "R"
        g = ctx["gL"] if side == "L" else ctx["gR"]
        x = ctx["xL"] if side == "L" else ctx["xR"]
        tt = ctx["tL"] if side == "L" else ctx["tR"]
        n = int(g.n)
        if n < 4:
            return None, {"跳过": "实体过少"}

        B = min(self.anchor_cap, int(self.batch), n)
        ids = torch.randperm(n, device=dev)[:B]

        Z = self.enc(x, *tt, n)                       # 在线端（有梯度）
        with torch.no_grad():
            Zm = self.enc_mom(x, *tt, n)              # 目标端（无梯度）
        q = F.normalize(Z[ids], dim=1)
        with torch.no_grad():
            k = F.normalize(Zm[ids], dim=1)

        # 正例 = 同一批实体在**目标端**的表示（官方 `pos_2 = self._model(pos_token)`）。
        # RSM 把正的相似度视为其上界 1/τ（式 2），两者在本文的短程训练下数值接近。
        l_pos = (q * k).sum(1, keepdim=True) / self.tau

        # 负例：**自负采样** —— 只从**本图**内取负。批内其余实体先作负例（去掉对角）。
        # 注意**不能 detach**：批内负例是 `q`（在线端、有梯度）的函数，detach 会把
        # 最主要的梯度来源切掉，只剩队列那一项。
        eye = torch.eye(B, device=dev, dtype=torch.bool)
        sim_in = (q @ k.t()).masked_fill(eye, -1e4)
        negs = [sim_in / self.tau]
        qd = self._queue[side]
        if qd:
            Q = torch.cat(qd, 0)
            negs.append((q @ Q.t()) / self.tau)
        l_neg = torch.cat(negs, 1)
        logits = torch.cat([l_pos, l_neg], 1)         # 第 0 列即正例 ⇒ label 全 0
        loss = F.cross_entropy(logits, torch.zeros(B, device=dev, dtype=torch.long))
        self.n_neg_last = int(l_neg.shape[1])

        # 入队（detach；队列长度按"批"计，超长则弹出最旧的一批）
        qd.append(k.detach())
        while len(qd) > self.neg_batches:
            qd.pop(0)
        self.step_log.append((ep, side, int(B), int(l_neg.shape[1])))
        return loss, {"锚点": int(B), "负例": int(l_neg.shape[1])}

    # ---------------------------------------------------------------- 记录
    def param_detail(self) -> dict:
        d = super().param_detail()
        d["负例队列(批)"] = int(self.neg_batches)
        return d

    def flags(self) -> dict:
        f = super().flags()
        f["编码器层数"] = 1
        f["温度 τ"] = float(self.tau)
        f["负例数(末次)"] = int(self.n_neg_last)
        return f
