# -*- coding: utf-8 -*-
"""Dual-AMN 重实现（WWW 2021，Mao et al., pp. 821–832, DOI 10.1145/3442381.3449897）。

**方法身份**：**有监督**实体对齐 —— 损失只在种子对齐对上计算（原文用 30% 预对齐对训练）。

**忠实实现的构件**（逐条对应 arXiv 2103.15452）
1. **图内关系感知层**（式 2–4）：
   $\\bm{h}^{l+1}_{e_i}=\\tanh\\big(\\sum_{e_j\\in\\mathcal N_{e_i}}\\sum_{r_k}\\alpha^{l}_{ijk}
   (\\bm{h}^{l}_{e_j}-2\\bm{h}_{r_k}^{\\!\\top}\\bm{h}^{l}_{e_j}\\bm{h}_{r_k})\\big)$
   —— 用**关系反射投影**（relational projection）替代线性变换 $W$，不增加参数；
   注意力 $\\alpha^{l}_{ijk}\\propto\\exp(\\bm v^{\\top}\\bm h_{r_k})$ 只依赖关系（meta-path softmax，
   即"在邻居的所有边型中做 softmax"）。多跳嵌入按式(4) 拼接
   $\\bm h^{multi}_{e_i}=[\\bm h^{0}\\|\\bm h^{1}\\|\\dots\\|\\bm h^{l}]$。
2. **代理匹配注意力**（式 7–8，跨图层）：用一组可学习 **proxy 向量** $\\{q_j\\}$ 表示对齐关系，
   $\\beta_{ij}=\\mathrm{softmax}_j\\cos(\\bm h^{multi}_{e_i},\\bm q_j)$，
   $\\bm h^{p}_{e_i}=\\sum_j\\beta_{ij}(\\bm h^{multi}_{e_i}-\\bm q_j)$。
   原文的卖点正在此处：把跨图交互的复杂度由 $O(|E_1||E_2|)$ 降到 $O(|E_1|+|E_2|)$。
   实现上利用 $\\sum_j\\beta_{ij}=1$ 化简为 $\\bm h^{multi}_{e_i}-\\bm\\beta_i\\bm Q$，
   避免物化 $(n,P,d)$ 的三维张量。
3. **门控聚合**（式 9–10）：$\\bm\\eta=\\sigma(\\bm M\\bm h^p+\\bm b)$，
   $\\bm h^{final}=\\bm\\eta\\odot\\bm h^{p}+(1-\\bm\\eta)\\odot\\bm h^{multi}$
   —— 融合"单图内"与"跨图"两路信息（此即 "dual" 所指）。
4. **归一化难样本挖掘损失**（式 15–19）：以 LogSumExp 平滑近似 TUNS 的"取最难的负例"，
   并对每条锚点样本的损失做**标准化**（减均值、除标准差，$\\mu,\\sigma^2$ **不参与反向传播**）。
5. **随机初始化的实体嵌入**：原文 basic 版本用 He 初始化 $H^e$（仅用关系结构，
   不用名/描述特征）—— 故本实现 **不使用 LaBSE**，与 ICL／SelfKG 的输入设定不同，
   这一点在表 6 与表 21 表注中如实声明。
6. **隐式对齐**：两图共用同一套变换（编码层、代理、门控），$\\bm h^{final}$ 天然在同一空间，
   原文无显式对齐步骤 ⇒ 本实现令 $W=I$（`align_mode="identity"`）。

**简化项（如实披露）**
- 式(15) 的负例集在对侧**全图**（$\\sum_{e_j'\\in E_2}$）；本实现以**批内负例**近似
  （每步取 `min(batch, |seeds|)` 个种子对，负例即批内其余对侧实体）。
- 式(16) 的 $\\sqrt{\\sigma^2-\\epsilon}$ 按原文实现（$\\epsilon=10^{-8}$）；分母用
  $\\sigma$ 而非 $\\sigma^2$，因为式(16) 的分子 $l_o-\\mu$ 与 $\\sigma$ 同量纲。
- 原文各图独立的 $H^r$ 规模与 proxy 个数未在正文给出；本实现取关系嵌入维 $=d$、
  proxy 个数 $=64$（超参搜索空间里的取值，非原文数值）。
- 测试阶段原文用 **CSLS** 缓解 hubness；本实现按本文统一契约改用**余弦**
  （`score_block` 与其余方法逐字符同式），属打分口径统一。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._gnn import _RecentBase, scatter_softmax


class _RelReflectLayer(nn.Module):
    """式(2)(3)：关系反射投影 + meta-path 注意力（Dual-AMN 的图内编码层）。"""

    def __init__(self, d: int, n_rel: int, drop: float = 0.1):
        super().__init__()
        # 式(2) 里 $\\bm h_{r_k}^{\\top}\\bm h_{e_j}$ 要求关系嵌入与实体嵌入**同维**
        self.rel_emb = nn.Embedding(max(int(n_rel), 1), d)
        self.v = nn.Parameter(torch.zeros(d))           # 式(3) 的注意力向量
        nn.init.normal_(self.rel_emb.weight, std=0.1)
        nn.init.normal_(self.v, std=0.1)
        self.drop = nn.Dropout(drop)

    def forward(self, h, src, dst, rel, n):
        hr = self.rel_emb(rel)                          # (E, d)
        hj = h[dst]                                     # (E, d)
        # 关系反射：h_j - 2 (h_r^T h_j) h_r —— 原文 "Relational Projection"，不引入 W
        proj = hj - 2.0 * (hr * hj).sum(-1, keepdim=True) * hr
        # α 只依赖关系（meta-path softmax）：在**中心节点**的所有入边上归一
        a = scatter_softmax((hr * self.v).sum(-1, keepdim=True), src, int(n))
        out = torch.zeros((int(n), h.shape[1]), device=h.device, dtype=h.dtype)
        out = out.index_add_(0, src, proj * a)
        return self.drop(torch.tanh(out))


class DualAMNEncoder(nn.Module):
    """Dual Attention Matching Network 的编码器。

    逐图持有实体嵌入表（两份，索引空间独立），**其余变换全部共享**（关系嵌入、注意力
    向量、proxy、门控）—— 共享是"两图落在同一空间"的保证，也是 `W=I` 的前提。
    """

    def __init__(self, nL: int, nR: int, d: int, n_rel: int, n_layer: int = 2,
                 n_proxy: int = 64, drop: float = 0.1):
        super().__init__()
        self.d, self.n_layer = int(d), max(int(n_layer), 1)
        self.emb = nn.ModuleDict({"L": nn.Embedding(max(int(nL), 1), d),
                                  "R": nn.Embedding(max(int(nR), 1), d)})
        for e in self.emb.values():                     # 原文用 He_initializer 随机初始化
            nn.init.kaiming_uniform_(e.weight, a=math.sqrt(5))
        self.layers = nn.ModuleList([_RelReflectLayer(d, n_rel, drop)
                                     for _ in range(self.n_layer)])
        dm = (self.n_layer + 1) * d                     # 多跳拼接后的维数（式 4）
        self.proxy = nn.Parameter(torch.randn(int(n_proxy), dm) * 0.05)
        self.gate = nn.Linear(dm, dm)                   # 式(9) 的 M, b

    def _multi_hop(self, side, src, dst, rel, n):
        h = self.emb[side].weight                       # 整张实体嵌入表即 h^0
        hs = [h]
        for L in self.layers:
            h = L(h, src, dst, rel, n)
            hs.append(h)
        return torch.cat(hs, -1)                        # h^multi（式 4）

    def forward(self, x, src, dst, rel, n, side="L"):
        H = self._multi_hop(side, src, dst, rel, n)
        beta = torch.softmax(F.normalize(H, dim=1) @ F.normalize(self.proxy, dim=1).t(),
                             dim=1)                     # 式(7)
        hp = H - beta @ self.proxy                      # 式(8)，用 Σ_j β_ij = 1 化简
        eta = torch.sigmoid(self.gate(hp))              # 式(9)
        return eta * hp + (1.0 - eta) * H               # 式(10)

    # ---- 参数分解（供表 17 的口径：编码器参数 与 逐实体嵌入表 分开计） ----
    def parts(self) -> dict:
        emb = int(sum(e.weight.numel() for e in self.emb.values()))
        rel = int(sum(L.rel_emb.weight.numel() for L in self.layers))
        vv = int(sum(L.v.numel() for L in self.layers))
        return {"实体嵌入表": emb, "关系嵌入": rel, "注意力向量 v": vv,
                "代理向量": int(self.proxy.numel()),
                "门控 M,b": int(sum(p.numel() for p in self.gate.parameters()))}


class DualAMN(_RecentBase):
    """Dual-AMN（有监督）—— 见模块 docstring 的忠实/简化清单。"""

    name = "Dual-AMN（WWW 2021，重实现）"
    impl = "reimplemented-from-paper"
    uses_seeds_for_align = True         # 有监督：损失只在种子对上计算
    uses_labse = False                  # 原文 basic 版本用随机初始化，不用名/描述特征
    uses_momentum = False               # 无 MoCo 结构 ⇒ 不复制一份编码器
    index_input = True                  # 编码器自带逐实体嵌入表
    align_mode = "identity"             # 共享变换 ⇒ 隐式对齐（原文无显式对齐步骤）

    def __init__(self, n_proxy: int = 64, lam: float = 30.0, mu_n: float = 10.0,
                 gamma: float = 1.0, eps: float = 1e-8, **kw):
        super().__init__(
            input_tag="随机初始化（原文 basic 版本：He 初始化实体嵌入，仅用关系结构）",
            **kw)
        self.n_proxy = int(n_proxy)
        self.lam, self.mu_n, self.gamma, self.eps = lam, mu_n, gamma, eps
        self.pair_log: list = []

    # ---------------------------------------------------------------- 编码器
    def _make_encoder(self, d_in, n_rel, nL, nR, dev):
        return DualAMNEncoder(nL, nR, self.d, n_rel, self.n_layer,
                              self.n_proxy, self.drop).to(dev)

    def _encode(self, x, t, n, side):
        return self.enc(x, *t, n, side=side)

    # ---------------------------------------------------------------- 损失
    def _nhsm(self, za, zb):
        """归一化难样本挖掘损失（式 15–19）的一个方向。

        `sim` 按式(20) 取**距离** $\\|h_i-h_j\\|_2^2$，故 $l_o=\\gamma+\\mathrm{sim}(a,p)-\\mathrm{sim}(a,n)$
        （越差越大）。负例集这里取**批内**（含正例自身：$l_o=\\gamma$，与式(18) 的
        "对全 $E_2$ 求和"在批内的对应物一致）。
        """
        d2 = torch.cdist(za, zb) ** 2                    # (B,B) 平方距离
        pos = d2.diagonal()                              # 正例距离（对角）
        # **符号**：sim 取距离 ⇒ 越大越差，故 l_o = γ + sim(pos) − sim(neg)
        # = γ + pos[:,None] − d2。写成 γ + d2 − pos 会把损失翻号（实测：Hits@1 塌到 0）。
        lo = self.gamma + pos[:, None] - d2              # 式(17)；j=正例时为 γ
        mu = lo.mean(1, keepdim=True).detach()           # 式(18)（不参与反向传播）
        var = lo.var(1, unbiased=False, keepdim=True).detach()   # 式(19)
        sd = (var - self.eps).clamp_min(1e-12).sqrt()    # 式(16) 的 sqrt(σ²-ε)
        ln = (lo - mu) / sd                              # 式(16)
        # log[1 + Σ_j exp(λ·l_n + τ)] ≡ logsumexp([0, λ·l_n + τ])
        z = torch.zeros_like(ln[:, :1])
        lse = torch.logsumexp(torch.cat([z, self.lam * ln + self.mu_n], 1), dim=1)
        return lse.mean()

    def _objective(self, ctx):
        dev, pair = ctx["dev"], ctx["pair"]
        seeds = getattr(pair, "seeds", None)
        if seeds is None or len(seeds) < 4:
            return None, {"跳过": "无种子"}
        S = int(len(seeds))
        B = min(int(self.batch), S)
        sel = torch.randperm(S, device=dev)[:B]
        s = torch.as_tensor(np.asarray(seeds, dtype=np.int64), device=dev)[sel]

        ZL = self._encode(ctx["xL"], ctx["tL"], int(ctx["gL"].n), "L")
        ZR = self._encode(ctx["xR"], ctx["tR"], int(ctx["gR"].n), "R")
        zl = F.normalize(ZL[s[:, 0]], dim=1)
        zr = F.normalize(ZR[s[:, 1]], dim=1)
        loss = self._nhsm(zl, zr) + self._nhsm(zr, zl)   # 式(15) 的两个方向
        self.pair_log.append((int(ctx["ep"]), int(B)))
        return loss, {"锚点对": int(B)}

    # ---------------------------------------------------------------- 参数
    def n_params(self) -> int:
        """编码器参数（**不含**逐实体嵌入表 —— 与表 17 的口径一致，避免与
        `n_embed_params()` 重复计数）。"""
        if self.enc is None:
            return 0
        return int(sum(p.numel() for p in self.enc.parameters())
                   - self.enc.parts()["实体嵌入表"])

    def n_embed_params(self) -> int:
        return 0 if self.enc is None else int(self.enc.parts()["实体嵌入表"])

    def param_detail(self) -> dict:
        if self.enc is None:
            return {"编码器参数": 0}
        d = dict(self.enc.parts())
        d["编码器参数（不含嵌入表）"] = self.n_params()
        d["多跳拼接维数"] = int((self.n_layer + 1) * self.d)
        return d

    def flags(self) -> dict:
        f = super().flags()
        f["代理个数"] = int(self.n_proxy)
        f["损失超参"] = f"γ={self.gamma:g}, λ={self.lam:g}, τ={self.mu_n:g}"
        return f
