# -*- coding: utf-8 -*-
"""ICL / ICLEA 重实现（CIKM 2022，Zeng et al., pp. 2465–2475, DOI 10.1145/3511808.3557364）。

**方法身份**：自监督实体对齐 —— 不使用任何种子对齐，靠"关系感知邻域聚合 + 交互式
对比学习 + 训练中挖掘伪对齐对"把两张 KG 对齐到同一空间。

**忠实实现的构件**
1. 关系感知邻域聚合（RGAT 精神）：`RelAwareLayer` 的关系门控注意力（见 `_gnn.py`）。
2. **交互式对比学习**：双向 InfoNCE，正例取自**训练中动态重挖**的伪对齐对。
3. **伪对齐对挖掘**：跨图互近邻（`mine_mutual`），每 `mine_every` 轮从当前嵌入重挖
   —— 这正是 "interactive" 的来源。
4. 动量编码器（EMA）提供 key，稳定对比目标。

**简化项（如实披露，正文表注须据此声明）**
- 原文的 **normalized hard sample mining** 重加权式未逐字复刻：以标准 InfoNCE 的批内
  负例近似；`tau` 与批大小即为此处的调节旋钮。
- 原文用 **Faiss 全库检索**做伪对挖掘（Top-1/双向 + 平方 L2 阈值），此处以分块
  互近邻 + 无阈值替代（云端无 faiss，2026-09-30 探针实测）。
- 原文的**描述文本**特征：本文 loader 未载入 DBP15K 的 description 字段，故以
  LaBSE(**实体名**) 单路替代（`flags()["输入特征"]` 会如实标出实际来源）。
- 最终对齐由**伪对上的正交 Procrustes**给出（原文以对比目标隐式对齐）；
  `flags()["对齐"]` 会标出实际用了哪一种。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ._gnn import _RecentBase, info_nce, mine_mutual


class ICL(_RecentBase):
    """ICLEA（自监督）—— 见模块 docstring 的忠实/简化清单。"""

    name = "ICL/ICLEA（CIKM 2022，重实现）"
    impl = "reimplemented-from-paper"
    uses_seeds_for_align = False        # 自监督：训练与对齐都**不碰 pair.seeds**

    def __init__(self, mine_every: int = 5, warmup_mine: bool = True, **kw):
        super().__init__(**kw)
        self.mine_every = int(mine_every)
        self.warmup_mine = bool(warmup_mine)
        self._pL = self._pR = None
        self.mine_log: list = []
        self.pair_history: list = []

    # ---------------------------------------------------------------- 目标
    def _current_pairs(self, ctx):
        """伪对齐对：首轮从**输入特征**挖（冷启动），之后每 `mine_every` 轮从嵌入重挖。"""
        ep = ctx["ep"]
        need = (self._pL is None) or (self.mine_every > 0 and ep % self.mine_every == 0)
        if not need:
            return self._pL, self._pR
        dev = ctx["dev"]
        if self._pL is None and not self.warmup_mine:
            return None, None
        with torch.no_grad():
            # **用在线编码器挖**，不用动量编码器：动量端在短程训练里必然滞后，
            # 从滞后端挖出的伪对会把错误正例喂给对比目标（实测：动量端只挖到约
            # 3.5k 对，而在线段能挖到约 2.4 万对）。动量端只承担 InfoNCE 的 key 侧。
            ZL = self.enc(ctx["xL"], *ctx["tL"], ctx["gL"].n).float()
            ZR = self.enc(ctx["xR"], *ctx["tR"], ctx["gR"].n).float()
        # 张量进、张量出 —— 挖掘全程留在设备上（见 `_gnn.mine_mutual` 的性能注记）
        iL, iR = mine_mutual(ZL, ZR)
        if len(iL) < 8:                                     # 互近邻过少时放宽
            iL, iR = mine_mutual(ZL, ZR, chunk=256)
        self._pL, self._pR = iL, iR
        self.mine_log.append((int(ep), int(len(iL))))
        return self._pL, self._pR

    def _objective(self, ctx):
        dev = ctx["dev"]
        iL, iR = self._current_pairs(ctx)
        if iL is None or len(iL) < 2:
            return None, {"伪对": 0}
        # 采一批伪对（不足则整取）
        n = len(iL)
        sel = (torch.randperm(n, device=dev)[:self.batch] if n > self.batch
               else torch.arange(n, device=dev))
        sL, sR = iL[sel], iR[sel]

        # 在线编码器（有梯度）
        ZL = self.enc(ctx["xL"], *ctx["tL"], ctx["gL"].n)
        ZR = self.enc(ctx["xR"], *ctx["tR"], ctx["gR"].n)
        # 动量编码器（无梯度）—— 提供对侧 key
        with torch.no_grad():
            ZLm = self.enc_mom(ctx["xL"], *ctx["tL"], ctx["gL"].n)
            ZRm = self.enc_mom(ctx["xR"], *ctx["tR"], ctx["gR"].n)

        qL = F.normalize(ZL[sL], dim=1)
        qR = F.normalize(ZR[sR], dim=1)
        kL = F.normalize(ZLm[sL], dim=1)
        kR = F.normalize(ZRm[sR], dim=1)

        # 双向 InfoNCE：L→R 与 R→L。批内其余伪对即负例（原文用全库负例 + 硬样本挖掘）。
        loss = info_nce(qL, kR, self.tau) + info_nce(qR, kL, self.tau)
        self.pair_history.append((int(ctx["ep"]), int(len(sel))))
        return loss, {"伪对": int(n), "批": int(len(sel))}

    # ---------------------------------------------------------------- 记录
    def param_detail(self) -> dict:
        d = super().param_detail()
        d["伪对齐对(末次)"] = int(len(self._pL)) if self._pL is not None else 0
        return d

    def flags(self) -> dict:
        f = super().flags()
        f["伪对挖掘轮次"] = int(self.mine_every)
        f["伪对数量轨迹"] = str(self.mine_log[-3:]) if self.mine_log else "[]"
        return f
