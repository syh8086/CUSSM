# -*- coding: utf-8 -*-
"""B 轨（近五年结构侧方法）共享算子 —— 方案见
`.workbuddy/repro_plan_trackB_2026-09-30.md`（路线乙：契约内重实现）。

**为什么自带一套 GNN 算子**：云端 venv（Py3.12 + torch 2.14+cu126）**没有 PyG/DGL**
（2026-09-30 只读探针实测），且本文纪律要求方法之间的差别只来自方法本身。稀疏聚合
直接用 `torch.index_add_` / `Tensor.index_reduce_('amax')` 实现，零额外依赖。

**硬纪律（写死在这里，供审稿追溯）**
1. 只读 `pair.left/right.rel_triples`、`pair.seeds`、`pair.val`；**绝不读 `pair.test`**。
2. 不使用各方法自带的候选截断与评测脚本；排名一律交给 `metrics.core.evaluate`。
3. 打分统一由子类 `score_block` 以 `(ZLs[left_idx] @ ZRs.T)` 给出 —— 与
   `_TrainableBase.score_block` **逐字符同式**，故候选池与并列裁决口径一致。
4. 输入特征按各方法原文设定；取不到时可回退本文字面视图，但**必须在 `flags()` 里
   如实标出实际用了哪一份**（表注据此声明）。
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

# `index_reduce_` 目前仍是 beta，PyTorch 会逐次打 UserWarning 刷屏（实测每步一条）。
# 这里只静音这一条 —— 不改行为，只为让远端日志可读。
warnings.filterwarnings("ignore", category=UserWarning, message=".*index_reduce.*")

from cussm.model import BaseMatcher, _rownorm, ortho_procrustes
from cussm.torch_backend import get_device


# ================================================================ 图结构
class Graph:
    """把 `KG` 的关系三元组整理成消息传递所需的**双向边**索引。

    反向边给一个独立 relation id（`r + n_rel`）—— 与 RGAT 家族一致，使"入边"与
    "出边"不共享同一套关系参数。
    """

    __slots__ = ("n", "n_rel", "src", "dst", "rel", "deg")

    def __init__(self, kg):
        T = np.asarray(kg.rel_triples, dtype=np.int64).reshape(-1, 3)
        nr = max(int(kg.n_rel), 1)
        self.n, self.n_rel = int(kg.n), nr
        if len(T):
            h, r, t = T[:, 0], T[:, 1], T[:, 2]
            self.src = np.concatenate([h, t])
            self.dst = np.concatenate([t, h])
            self.rel = np.concatenate([r, r + nr])
        else:                                   # 空图（防御：某些自检集可能无关系三元组）
            self.src = self.dst = self.rel = np.zeros(0, dtype=np.int64)
        self.deg = np.bincount(self.src, minlength=self.n).astype(np.float32)

    def tensors(self, device):
        return (torch.as_tensor(self.src, dtype=torch.long, device=device),
                torch.as_tensor(self.dst, dtype=torch.long, device=device),
                torch.as_tensor(self.rel, dtype=torch.long, device=device))


# ================================================================ 稀疏算子
def scatter_softmax(e: torch.Tensor, index: torch.Tensor, n: int,
                    eps: float = 1e-16) -> torch.Tensor:
    """按 `index`（目标节点）对逐边分数 `e`（E,1）做 softmax。

    等价于 `softmax` 的 scatter 版：先散射取每节点最大值（`index_reduce_('amax')`），
    减去后取指数，再散射求和归一。不用 `scatter_reduce` 的 sum 分支是因为
    需要两次不同的归约（amax 与 sum），`index_*` 两者都支持。
    """
    e = e.float()
    mx = torch.full((n, 1), -1e30, device=e.device, dtype=e.dtype)
    mx = mx.index_reduce_(0, index, e.detach(), "amax", include_self=True)
    ex = torch.exp(e - mx[index].detach())
    s = torch.zeros((n, 1), device=e.device, dtype=e.dtype)
    s = s.index_add_(0, index, ex.detach())
    return ex / (s[index] + eps)


class RelAwareLayer(nn.Module):
    """一层**关系感知邻域聚合**（RGAT 风格，单头）。

        e_ij = <W_q h_i , W_k h_j + W_rel r_ij> / sqrt(d)
        a_ij = softmax_{j in N(i)} e_ij
        h_i' = act( W_self h_i + Σ_j a_ij · W_v h_j )

    ICL/ICLEA、Dual-AMN、RNHGT 三者的差异在**上一层网络怎么叠、目标函数怎么定义**，
    邻域聚合的共同骨架即此式（各自的门控/路径项由子类网络的额外分支承载）。
    """

    def __init__(self, d_in: int, d_out: int, n_rel: int, d_rel: int = 64,
                 drop: float = 0.1):
        super().__init__()
        self.d_out = int(d_out)
        self.W_self = nn.Linear(d_in, d_out, bias=False)
        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)
        self.rel_emb = nn.Embedding(max(int(n_rel), 1), d_rel)
        self.W_rel = nn.Linear(d_rel, d_out, bias=False)
        self.drop = nn.Dropout(drop)
        self.scale = float(max(d_out, 1)) ** 0.5
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, h, src, dst, rel, n):
        q = self.W_q(h)[src]
        k = self.W_k(h)[dst] + self.W_rel(self.rel_emb(rel))
        v = self.W_v(h)[dst]
        e = (q * k).sum(-1, keepdim=True) / self.scale
        a = scatter_softmax(e, src, n)
        msg = v * a
        out = torch.zeros((n, self.d_out), device=h.device, dtype=h.dtype)
        out = out.index_add_(0, src, msg)
        out = out + self.W_self(h)
        return self.drop(torch.relu(out))


class RelAwareEncoder(nn.Module):
    """把输入特征投影到 d 维后叠 `n_layer` 层关系感知聚合。"""

    def __init__(self, d_in: int, d: int, n_rel: int, n_layer: int = 2,
                 d_rel: int = 64, drop: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [RelAwareLayer(d if i else d, d, n_rel, d_rel, drop)
             for i in range(n_layer)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x, src, dst, rel, n):
        h = torch.relu(self.proj(x))
        for L in self.layers:
            h = L(h, src, dst, rel, n)
        return self.norm(h)


class PlainGATLayer(nn.Module):
    """**无关系信息**的单头图注意力层 —— SelfKG 原文的构件。

    原文（arXiv 2203.01044）明写："we directly use a **single-head graph attention
    network with one layer** to aggregate pre-trained embeddings of one-hop
    neighbors"，且其消融发现"multi-hop information actually harms the performance"。

    因此 SelfKG 的编码器与本文其余方法**不同**：不注入关系嵌入，且层数固定为 1。
    式与 `RelAwareLayer` 同源，只去掉 `W_rel(rel_emb(rel))` 一项，便于对照。
    """

    def __init__(self, d_in: int, d_out: int, drop: float = 0.1):
        super().__init__()
        self.W_self = nn.Linear(d_in, d_out, bias=False)
        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)
        self.drop = nn.Dropout(drop)
        self.scale = float(max(d_out, 1)) ** 0.5

    def forward(self, h, src, dst, rel=None, n=None):
        q = self.W_q(h)[src]
        k = self.W_k(h)[dst]
        v = self.W_v(h)[dst]
        e = (q * k).sum(-1, keepdim=True) / self.scale
        a = scatter_softmax(e, src, int(n))
        out = torch.zeros((int(n), self.W_v.out_features), device=h.device, dtype=h.dtype)
        out = out.index_add_(0, src, v * a)
        return self.drop(torch.relu(out + self.W_self(h)))


class PlainGATEncoder(nn.Module):
    """SelfKG 的编码器：输入特征 → d 维 → `n_layer` 层无关系图注意力。

    **`n_layer` 默认 1 且不建议改**（见 `PlainGATLayer` 的注记）；调用方若传别的值，
    会被 `SelfKG.__init__` 覆盖并在 `flags()` 里记下实际层数。
    """

    def __init__(self, d_in: int, d: int, n_layer: int = 1, drop: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [PlainGATLayer(d, d, drop) for _ in range(max(int(n_layer), 1))])
        self.norm = nn.LayerNorm(d)

    def forward(self, x, src, dst, rel=None, n=None):
        h = torch.relu(self.proj(x))
        for L in self.layers:
            h = L(h, src, dst, rel, n)
        return self.norm(h)


# ================================================================ 输入特征
_LABSE_CACHE: dict = {}


def _labse_load(mid: str, device):
    """按 `(模型 id, 设备)` 缓存 —— 同一次实验里两侧、多个方法都要同一份权重，
    fr_en 上重复 `from_pretrained` 会白读 ~1.8 GB 数遍（实测为瓶颈）。"""
    key = (mid, str(device))
    if key not in _LABSE_CACHE:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(mid)
        mdl = AutoModel.from_pretrained(mid).to(device).eval()
        for p in mdl.parameters():
            p.requires_grad_(False)
        _LABSE_CACHE[key] = (tok, mdl)
    return _LABSE_CACHE[key]


def _labse_texts(texts, device, batch: int = 64, max_len: int = 64):
    """LaBSE 句子嵌入（`setu4993/LaBSE`，均值池化 + L2 归一）。

    云端已确证 `hf-mirror.com` 可达（`HF_ENDPOINT` 由环境给出）；`huggingface.co`
    不通，故**必须**走镜像。取不到时抛异常，由 `name_features` 回退。
    """
    mid = os.environ.get("CUSSM_LABSE_ID", "setu4993/LaBSE")
    tok, mdl = _labse_load(mid, device)
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            b = [(t if str(t).strip() else "<empty>") for t in texts[i:i + batch]]
            enc = tok(b, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            last = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            emb = (last * m).sum(1) / m.sum(1).clamp(min=1e-6)
            outs.append(emb.float().cpu())
    return torch.cat(outs, 0).numpy().astype(np.float32)


def name_features(kg, d_sem: int = 256, seed: int = 2026, device=None,
                  use_labse: bool = True):
    """实体名特征 —— 返回 `(feats, source_tag)`。

    优先 LaBSE（跨语言，正是 ICLEA 原文的输入）；不可用则回退本文 `build_view` 的
    表层名块。**`source_tag` 必须进 `flags()`**，正文据此如实声明"输入特征替换"。
    """
    dev = get_device("auto") if device is None else device
    if use_labse and os.environ.get("CUSSM_NO_LABSE", "") not in ("1", "true", "yes"):
        try:
            X = _labse_texts(kg.surfaces(), dev)
            return X, "LaBSE(setu4993/LaBSE)"
        except Exception as exc:                     # noqa: BLE001
            print(f"    [name] LaBSE 不可用（{type(exc).__name__}: {exc}），回退本文字面视图",
                  flush=True)
    from cussm.features import build_view
    v = build_view(kg, 64, d_sem, seed)
    Z = v.sem_name if v.sem_name is not None else v.sem
    return np.asarray(Z, dtype=np.float32), "本文字面视图(build_view.sem_name)"


# ================================================================ 对比目标与挖掘
def info_nce(q: torch.Tensor, k: torch.Tensor, tau: float = 0.05,
             queue: torch.Tensor | None = None) -> torch.Tensor:
    """对称 InfoNCE。`q`,`k` 已 L2 归一，第 i 行互为正例；`queue` 为额外负例（已归一）。"""
    keys = k if queue is None else torch.cat([k, queue], 0)
    logits = (q @ keys.t()) / tau
    lab = torch.arange(q.shape[0], device=q.device)
    return 0.5 * (nn.functional.cross_entropy(logits, lab)
                  + nn.functional.cross_entropy(logits.t()[:q.shape[0]], lab))


def mine_mutual(ZL, ZR, chunk: int | None = None):
    """跨图**互近邻**挖掘伪对齐对（ICL 的 pseudo-pair mining）。

    分块算相似度，避免物化 `(nL, nR)` 稠密矩阵；返回 `(idx_L, idx_R)`。

    **传入 torch 张量则全程留在设备上**（GPU 上实测比 numpy 版快两个数量级：
    fr_en 的 66k×105k 相似度，numpy 每轮约 90 s，同为 fp32 的 T4 每轮约 0.6 s）。
    因此调用方**不要**先 `.cpu().numpy()` —— 那会把热路径又拖回 CPU。
    """
    if torch.is_tensor(ZL):
        return _mine_mutual_t(ZL, ZR, chunk)
    nL, nR = int(ZL.shape[0]), int(ZR.shape[0])
    if chunk is None:
        chunk = max(256, min(4096, int(2 ** 26 // max(nR, 1))))     # ~64M 元素/块
    bestR = np.empty(nL, dtype=np.int64)
    for i in range(0, nL, chunk):
        bestR[i:i + chunk] = (ZL[i:i + chunk] @ ZR.T).argmax(1)
    bestL = np.empty(nR, dtype=np.int64)
    for j in range(0, nR, chunk):
        bestL[j:j + chunk] = (ZR[j:j + chunk] @ ZL.T).argmax(1)
    left = np.arange(nL, dtype=np.int64)
    ok = bestL[bestR] == left
    return left[ok], bestR[ok]


def _mine_mutual_t(ZL, ZR, chunk: int | None = None):
    """`mine_mutual` 的设备端实现（进出都是 torch 张量，索引留在原设备）。"""
    dev = ZL.device
    nL, nR = int(ZL.shape[0]), int(ZR.shape[0])
    if chunk is None:                       # 目标 ~256 MB 的临时相似度块
        chunk = max(256, min(4096, (2 ** 26) // max(nR, 1)))
    with torch.no_grad():
        zl, zr = ZL.float(), ZR.float()
        bestR = torch.empty(nL, dtype=torch.long, device=dev)
        for i in range(0, nL, chunk):
            bestR[i:i + chunk] = (zl[i:i + chunk] @ zr.t()).argmax(1)
        bestL = torch.empty(nR, dtype=torch.long, device=dev)
        for j in range(0, nR, chunk):
            bestL[j:j + chunk] = (zr[j:j + chunk] @ zl.t()).argmax(1)
        left = torch.arange(nL, dtype=torch.long, device=dev)
        ok = bestL[bestR] == left
        return left[ok], bestR[ok]


# ================================================================ 基类骨架
class _RecentBase(BaseMatcher):
    """「建图 → 编码 → 对齐 → 打分」的统一骨架；子类只实现 `_objective`。

    `uses_seeds_for_align=True` 表示该方法**在原文中即用种子对齐**（Dual-AMN / RNHGT）；
    `False` 表示自监督（SelfKG / ICL）—— 此时训练与对齐都**不得触碰 `pair.seeds`**，
    以保持"无监督"这一方法身份；`flags()` 会把它标出来。
    """

    family = "结构嵌入（近五年）"
    impl = "reimplemented-from-paper"
    uses_seeds_for_align = True
    uses_labse = True
    # 是否配一个动量/目标编码器（ICL、SelfKG 需要；Dual-AMN 不需要 ⇒ 省一份参数）
    uses_momentum = True
    # True ⇒ 编码器自带逐实体嵌入表，`x` 只用来给出实体下标（Dual-AMN basic 的随机初始化）
    index_input = False
    # 对齐方式："procrustes"（在种子或伪对上做正交对齐）｜"identity"（统一空间内直接余弦，
    # 对应原文**隐式对齐**——SelfKG 的 uni-space 与 Dual-AMN 的共享代理都属此类）
    align_mode = "procrustes"

    def __init__(self, d: int = 256, n_layer: int = 2, d_rel: int = 64,
                 epochs: int = 60, lr: float = 1e-3, batch: int = 4096,
                 tau: float = 0.05, drop: float = 0.1, d_sem: int = 256,
                 seed: int = 2026, device: str = "auto",
                 max_minutes: float | None = None, weight_decay: float = 0.0,
                 ema: float = 0.9, input_tag: str | None = None, **kw):
        self.d, self.n_layer, self.d_rel = d, n_layer, d_rel
        self.epochs, self.lr, self.batch = epochs, lr, batch
        self.tau, self.drop, self.d_sem = tau, drop, d_sem
        self.seed, self.device = seed, device
        self.max_minutes, self.weight_decay = max_minutes, weight_decay
        # 动量编码器的 EMA 系数。**必须按步数定，不能照抄 MoCo 的 0.999**：
        # 本文每次 fit 只有 `epochs` 步（默认 60），0.999^60 ≈ 0.94 ⇒ 动量编码器
        # 几乎停在随机初始化上，用它挖伪对只能得到约 3.5k 对（实测），而训练好的
        # 在线编码器能挖到约 2.4 万对 —— 对比目标因此被喂了错误的正例。
        # 默认 0.9 使 0.9^60 ≈ 0.002，动量端能跟上；长跑时可上调。
        self.ema = float(ema)
        # `index_input=True` 时如实声明输入是什么（供 `flags()` 与正文表注引用）
        self.input_tag = input_tag or "?"
        self.train_time = 0.0
        self.epochs_done = 0
        self.steps_done = 0                 # 梯度步数（≈ epochs；表注据此声明"轮 vs 步"）
        self.ZLs = self.ZRs = None
        self.enc = None
        self.enc_mom = None
        self.feat_source = {"left": "?", "right": "?"}
        self.align_source = "?"

    # ---- 子类钩子：返回 (有梯度损失, 记录用 dict) ----
    def _objective(self, ctx) -> tuple:
        raise NotImplementedError

    # ---- 子类钩子：输入特征 / 编码器 / 前向 ----
    def _input_features(self, kg, dev):
        """默认：LaBSE 实体名（不可用则回退本文字面视图）。Dual-AMN 覆写为随机初始化。"""
        return name_features(kg, self.d_sem, self.seed, dev, self.uses_labse)

    def _make_encoder(self, d_in, n_rel, nL, nR, dev):
        return RelAwareEncoder(d_in, self.d, n_rel, self.n_layer,
                               self.d_rel, self.drop).to(dev)

    def _encode(self, x, t, n, side):
        """统一前向入口。`side ∈ {"L","R"}` 供编码器内部选择逐图的实体嵌入表。"""
        return self.enc(x, *t, n)

    # ---- 对齐 ----
    def _align(self, pair, ZL, ZR):
        """得到左→右的对齐映射 W（`ZL/ZR` 为**设备上**的 torch 张量，索引不出设备）。"""
        if self.align_mode == "identity":
            self.align_source = "隐式（两图共用同一套变换 ⇒ 统一空间内直接余弦，原文无显式对齐步骤）"
            return np.eye(int(ZL.shape[1]), dtype=np.float32)
        if self.uses_seeds_for_align and len(pair.seeds):
            self.align_source = "种子正交 Procrustes"
            return ortho_procrustes(ZL[pair.seeds[:, 0]].float().cpu().numpy(),
                                    ZR[pair.seeds[:, 1]].float().cpu().numpy())
        iL, iR = mine_mutual(ZL, ZR)
        if len(iL) < 8:                       # 兜底：互近邻过少时放宽块大小重挖
            iL, iR = mine_mutual(ZL, ZR, chunk=256)
        if len(iL) < 2:
            raise RuntimeError("自监督对齐挖不到任何伪对齐对，无法对齐（该配置下不可比）")
        self.align_source = f"自监督互近邻 Procrustes（{len(iL)} 对）"
        return ortho_procrustes(ZL[iL].float().cpu().numpy(),
                                ZR[iR].float().cpu().numpy())

    # ---- 拟合 ----
    def fit(self, pair):
        t0 = time.time()
        dev = get_device(self.device)
        torch.manual_seed(self.seed)

        gL, gR = Graph(pair.left), Graph(pair.right)
        if self.index_input:
            # 编码器自带逐实体嵌入表 ⇒ 输入只是一个"取整张表"的下标向量；
            # 特征来源如实记为 `input_tag`（Dual-AMN basic 用的是随机初始化）。
            xL = torch.arange(pair.left.n, device=dev)
            xR = torch.arange(pair.right.n, device=dev)
            self.feat_source = {"left": self.input_tag, "right": self.input_tag}
        else:
            XL, tagL = self._input_features(pair.left, dev)
            XR, tagR = self._input_features(pair.right, dev)
            self.feat_source = {"left": tagL, "right": tagR}
            xL = torch.as_tensor(_rownorm(XL), device=dev)
            xR = torch.as_tensor(_rownorm(XR), device=dev)
        tL, tR = gL.tensors(dev), gR.tensors(dev)

        # 关系表要覆盖**反向边**的 id：`Graph` 里正向用 0..nr-1、反向用 nr..2nr-1，
        # 故表长必须是 2×n_rel（曾漏乘 2 而抛 `IndexError: index out of range`）。
        n_rel = 2 * max(gL.n_rel, gR.n_rel)
        d_in = None if self.index_input else int(xL.shape[1])
        self.enc = self._make_encoder(d_in, n_rel, gL.n, gR.n, dev)

        # 共同空间：**一个共享编码器**作用在两张图上（GCN-Align / ICL 家族的做法），
        # 这样跨图余弦才在同一坐标系里。动量编码器按 ICL 的 momentum contrast 配。
        if self.uses_momentum:
            self.enc_mom = self._make_encoder(d_in, n_rel, gL.n, gR.n, dev)
            self.enc_mom.load_state_dict(self.enc.state_dict())
            for p in self.enc_mom.parameters():
                p.requires_grad_(False)

        opt = torch.optim.Adam(self.enc.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        budget = None if not self.max_minutes else float(self.max_minutes)
        ts = time.time()
        done = 0
        for ep in range(self.epochs):
            if budget and (time.time() - ts) / 60 > budget:
                break
            ctx = {"ep": ep, "xL": xL, "xR": xR, "tL": tL, "tR": tR,
                   "gL": gL, "gR": gR, "pair": pair, "dev": dev, "opt": opt}
            loss, _info = self._objective(ctx)
            if loss is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.enc.parameters(), 5.0)
                opt.step()
                self.steps_done += 1
            if self.enc_mom is not None:          # 动量更新（EMA 系数见 self.ema 的注释）
                with torch.no_grad():
                    for pm, po in zip(self.enc_mom.parameters(), self.enc.parameters()):
                        pm.mul_(self.ema).add_(po.detach(), alpha=1.0 - self.ema)
            done += 1
        self.epochs_done = done
        self.train_time = time.time() - t0

        with torch.no_grad():
            ZLt = self._encode(xL, tL, gL.n, "L").float()
            ZRt = self._encode(xR, tR, gR.n, "R").float()
            W = self._align(pair, ZLt, ZRt)          # 挖掘留在设备上
        ZL = ZLt.cpu().numpy()
        ZR = ZRt.cpu().numpy()
        self.W = W
        self.ZLs = _rownorm(ZL @ W)
        self.ZRs = _rownorm(ZR)
        return self

    # ---- 输出契约 ----
    def score_block(self, pair, left_idx):
        return (self.ZLs[left_idx] @ self.ZRs.T).astype(np.float32)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.enc.parameters())) if self.enc else 0

    def n_embed_params(self) -> int:
        return 0                                   # 无逐实体嵌入表（特征是输入）

    def param_detail(self) -> dict:
        d = {"编码器参数": self.n_params()}
        return d

    def flags(self) -> dict:
        return {"实现": self.impl,
                "输入特征": f"L={self.feat_source['left']} / R={self.feat_source['right']}",
                "对齐": self.align_source,
                "用种子": bool(self.uses_seeds_for_align),
                "完成轮数": int(self.epochs_done),
                "梯度步数": int(self.steps_done)}
