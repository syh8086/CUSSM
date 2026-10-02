# -*- coding: utf-8 -*-
"""统一数据契约（第 5.4.4 节 I/O 契约的代码实现）。

**唯一的输入口径**：无论哪个方法、哪条 track，都只吃一个 `Pair` 对象。
    Pair.left / Pair.right   : 两侧的 `KG`（实体、关系三元组、属性三元组）
    Pair.seeds / val / test  : 对齐监督的三段划分（下标对）
这样"基线方法"与"本文方法"的输入在字节级上是同一份数据，可比性不靠约定靠构造。

**唯一的输出口径**：每个方法只实现一个方法
    score_block(pair, left_idx) -> (len(left_idx), n_right) float32
即"给定左侧实体子集，给出对右侧全部候选的得分（越大越可能对齐）"。
分块是为了在 2 万 × 2 万 的候选矩阵上不吃满内存；排名的计算方式对所有方法完全相同。

**三个真实数据源**（均已通过 SHA-1 校验，见 data/manifest.json）：
    1. DBP15K fr_en / zh_en / ja_en  —— 跨语言 KG 实体对齐（Track B 主实验）
    2. Flickr30K Entities           —— 视觉区域 ↔ 文本短语对齐（Track A' 代理实验）
    3. countries_S1/S2/S3, wn18rr   —— 小规模 KG，用于单元自检与受控结构实验
"""
from __future__ import annotations

import os
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from urllib.parse import unquote

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw")

_LIT_RE = re.compile(r'^"(.*?)"(?:@([A-Za-z-]+)|\^\^<[^>]*>)?$', re.S)
_NT_RE = re.compile(r"^<([^>]+)>\s+<([^>]+)>\s+(.+?)\s*\.\s*$")
_WS = re.compile(r"\s+")


# ================================================================ 基本容器
@dataclass
class KG:
    """一个模态范畴（第 3 章定义 1 的代码对应物）。"""
    name: str
    ents: list[str]                       # 实体标识（URI 或 视觉单元 id）
    rel_triples: np.ndarray               # (m,3) int32  (头, 关系, 尾)
    att_triples: list[tuple]              # (实体下标, 属性键, 属性值文本, 语言标签)
    rel_names: list[str]                  # 关系键 → 名称
    e2i: dict
    surface: list[str] | None = None      # 实体的可读表层形式（用于语义路由）

    @property
    def n(self) -> int:
        return len(self.ents)

    @property
    def n_rel(self) -> int:
        return len(self.rel_names)

    def surfaces(self) -> list[str]:
        if self.surface is not None:
            return self.surface
        return [local_name(e) for e in self.ents]

    def __repr__(self) -> str:
        return (f"KG({self.name}: {self.n} 实体 / {self.n_rel} 关系 / "
                f"{len(self.rel_triples)} 关系三元组 / "
                f"{len(self.att_triples)} 属性断言)")

    __str__ = __repr__


@dataclass
class Pair:
    """一对模态范畴 + 对齐监督。所有方法与指标都只依赖这个对象。"""
    name: str
    left: KG
    right: KG
    seeds: np.ndarray                     # (n_s,2) int32 —— 训练用（种子对齐）
    val: np.ndarray                       # (n_v,2) int32 —— 选超参用（从 seeds 中切出）
    test: np.ndarray                      # (n_t,2) int32 —— 评测用（与 seeds 不交）
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.meta
        s = (f"{self.name}: 左侧 {self.left.n} 实体 / {self.left.n_rel} 关系 / "
             f"{len(self.left.att_triples)} 属性断言；"
             f"右侧 {self.right.n} 实体 / {self.right.n_rel} 关系 / "
             f"{len(self.right.att_triples)} 属性断言；"
             f"seeds {len(self.seeds)} / val {len(self.val)} / test {len(self.test)}")
        for k in ("truth_per_left", "unit", "note"):
            if k in m:
                s += f"；{k}={m[k]}"
        return s

    # dataclass 的默认 repr 会把 `ents` 整份实体表倾倒进日志（实测单次 print
    # 可产出近 10 MB），排查时反而看不到关键数字。改成摘要。
    def __repr__(self) -> str:
        return f"Pair({self.summary()})"

    __str__ = __repr__


# ================================================================ 工具
def local_name(uri: str) -> str:
    """取 URI 末段并做百分号解码、Unicode 规范化。"""
    s = uri.rstrip(">").rstrip("/")
    s = s.rsplit("/", 1)[-1]
    s = s.rsplit("#", 1)[-1]
    s = unquote(s)
    s = s.replace("_", " ")
    return _WS.sub(" ", unicodedata.normalize("NFC", s)).strip()


def _parse_literal(tok: str) -> tuple[str, str | None]:
    """从 N-Triples 字面量取出 (纯文本, 语言标签)。"""
    m = _LIT_RE.match(tok.strip())
    if m:
        return m.group(1), m.group(2)
    return tok.strip().strip('"'), None


def _strip_uri(u: str) -> str:
    return u.strip().lstrip("<").rstrip(">").strip()


def _read_triples(path: str, is_attr: bool):
    """读 DBP15K 三元组文件。

    实测两种格式并存（2026-09-27）：
      · `*_rel_triples`：**制表符**分隔、URI 不带尖括号 →  三字段直切；
      · `*_att_triples`：**空格**分隔的 N-Triples、URI 带尖括号、字面量内含空格
        → 必须按 `<s> <p> "…"@lang .` 解析，否则含空格的字面量会被截断。
    返回 (rel_or_att 列表, 关系名集合)。
    """
    rels, atts, rel_names = [], [], []
    seen_rel: dict[str, int] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if is_attr:
                m = _NT_RE.match(line.strip())
                if m:
                    s, p, obj = m.group(1), m.group(2), m.group(3)
                else:                                   # 退化：按空白切
                    q = line.split()
                    if len(q) < 3:
                        continue
                    s, p, obj = q[0], q[1], " ".join(q[2:])
                val, lang = _parse_literal(obj)
                if not val:
                    continue
                atts.append((_strip_uri(s), local_name(_strip_uri(p)), val, lang))
            else:
                parts = line.split("\t")
                if len(parts) < 3:
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                h, r, t = _strip_uri(parts[0]), _strip_uri(parts[1]), _strip_uri(parts[2])
                if r not in seen_rel:
                    seen_rel[r] = len(rel_names)
                    rel_names.append(r)
                rels.append((h, seen_rel[r], t))
    return rels, atts, rel_names


# ================================================================ 1. DBP15K
def _build_kg(dirpath: str, side: str, kg_name: str,
              keep_literal: str = "lang") -> tuple[KG, dict]:
    """构建单侧 KG，**独立下标空间**。

    实测确认（2026-09-27）：DBP15K 两侧 URI 命名空间完全不相交
    （en 侧 `http://dbpedia.org/resource/…`，fr 侧 `http://fr.dbpedia.org/resource/…`，
    各取 20 万行采样交集为 0），因此 left/right 必须各用一套实体下标；
    这一点与第 3 章"两个模态范畴"的设定天然吻合。

    `keep_literal` 控制属性断言的口径：
      · "lang"（默认）—— 只保留带语言标签的字符串字面量，即真正的**语义内容**；
      · "all"        —— 全保留（含 `"88.3"^^xsd:double` 这类数值字面量）。
    数值字面量跨语言几乎不携带区分信息，且会稀释 TF-IDF，故默认剔除。
    """
    rels_raw, _, rel_names = _read_triples(os.path.join(dirpath, f"{side}_rel_triples"),
                                           is_attr=False)
    _, atts_raw, _ = _read_triples(os.path.join(dirpath, f"{side}_att_triples"),
                                   is_attr=True)
    n_att_raw = len(atts_raw)
    n_num = sum(1 for a in atts_raw if a[3] is None)
    if keep_literal == "lang":
        atts_raw = [a for a in atts_raw if a[3] is not None]
    ent_ids: dict[str, int] = {}
    ent_list: list[str] = []

    def eid(u: str) -> int:
        u = u.strip()
        i = ent_ids.get(u)
        if i is None:
            i = len(ent_list)
            ent_ids[u] = i
            ent_list.append(u)
        return i

    for h, _, t in rels_raw:
        eid(h); eid(t)
    for h, _, _, _ in atts_raw:
        eid(h)
    tri = np.array([[eid(h), r, eid(t)] for h, r, t in rels_raw], dtype=np.int32) \
        if rels_raw else np.zeros((0, 3), np.int32)
    att = [(eid(h), k, v, lg) for h, k, v, lg in atts_raw]
    stats = {"n_att_raw": n_att_raw, "n_att_numeral_dropped": n_num,
             "n_att_kept": len(att), "n_rel_triples": int(tri.shape[0])}
    return KG(name=kg_name, ents=ent_list, rel_triples=tri, att_triples=att,
              rel_names=rel_names, e2i=ent_ids), stats


def load_dbp15k(lang: str = "fr_en", seed_ratio: float = 0.30,
                val_ratio_of_seeds: float = 0.10, seed: int = 2026,
                max_test: int | None = None) -> Pair:
    """加载 DBP15K 一个语向对，按标准协议切分种子/验证/测试。

    协议（对齐 OpenEA 与多数 EA 工作的做法）：
      · 候选集 = 目标 KG 的**全部实体**，不做任何预筛或截断；
      · 15,000 条 ILL 中 30% 作种子（其中 10% 再切为验证），70% 作测试；
      · 测试实体在训练期完全屏蔽。

    `max_test` 只限制**参与评测的查询数**（加快调试），不改变候选集 ——
    排名是在完整候选集上算的，因此指标口径不受影响。
    """
    d = os.path.join(RAW, f"dbp15k_{lang}")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"未找到 {d}，先运行 code/download_data.py --group dbp15k")
    other = lang.split("_")[0]

    left, st_l = _build_kg(d, other, f"DBP15K-{lang}:{other}")   # 非英语侧
    right, st_r = _build_kg(d, "en", f"DBP15K-{lang}:en")        # 英语侧

    ill = []
    with open(os.path.join(d, "ent_ILLs"), encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                p = line.split()
            if len(p) < 2:
                continue
            a, b = p[0].strip(), p[1].strip()
            # 文件格式为「非英语侧 \t 英语侧」
            if a.startswith("http://dbpedia.org/"):
                a, b = b, a
            if a in left.e2i and b in right.e2i:
                ill.append((left.e2i[a], right.e2i[b]))
    ill = np.array(ill, dtype=np.int32)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ill))
    n_seed = int(round(len(ill) * seed_ratio))
    s_idx, t_idx = perm[:n_seed], perm[n_seed:]
    n_val = max(1, int(round(n_seed * val_ratio_of_seeds)))
    val, seeds = ill[s_idx[:n_val]], ill[s_idx[n_val:]]
    test = ill[t_idx]
    if max_test and len(test) > max_test:
        test = test[np.sort(rng.choice(len(test), max_test, replace=False))]

    return Pair(name=f"DBP15K-{lang}", left=left, right=right,
                seeds=seeds, val=val, test=test,
                meta={"dataset": "DBP15K", "lang": lang, "seed_ratio": seed_ratio,
                      "seed": seed, "unit": "实体", "truth_per_left": 1,
                      "candidates": f"目标 KG 全部 {right.n} 个实体",
                      "n_ill": int(len(ill)),
                      "att_left": st_l, "att_right": st_r,
                      "note": "测试实体训练期屏蔽；两侧下标空间独立"})


# ================================================================ 2. Flickr30K Entities
# 官方句子标注格式是 `[/EN#<链号>/<类型>␣<短语>]` —— **类型与短语之间是空格**，
# 不是斜杠。原实现写成 `\[/EN#(\d+)/([a-zA-Z]+)/(.*?)\]`（多了第二个 `/`），
# 导致在真实数据上几乎不命中：实测 Sentences/1000092795.txt 中
# `\[/EN#\d+/[a-zA-Z]+` 命中 14 次，而再加一个 `/` 后命中 **0** 次；
# 全库 31,783 图只解析出 1,599 条链（真实应约 15 万条）。
# 2026-09-28 修正：分隔符改为「空白或斜杠」以兼容两种写法。
_ENT_RE = re.compile(r"\[/EN#(\d+)/([a-zA-Z]+)[\s/]+(.*?)\]")


# ---- Track A′ 的关系词汇表（2026-09-28 新增）----
# 左（视觉）侧：由边界框几何**确定性地**导出，只用坐标、不用链号，故不泄露真值。
LEFT_SPATIAL_RELS = ["above", "below", "left_of", "right_of", "overlaps"]
# 右（文本）侧：由句内次序与短语类型导出，同样不读对齐真值。
RIGHT_ORDER_RELS = ["precedes", "same_type"]

# 图像本体目录（由 hf-mirror 的 nlphuji/flickr30k 取得，2026-09-28）
F30K_IMG_DIRS = (
    os.path.join(RAW, "flickr30k", "flickr30k-images"),
    os.path.join(RAW, "flickr30k", "flickr30k_images"),
    os.path.join(RAW, "flickr30k", "images"),
)


def flickr_image_dir() -> str | None:
    """返回实际存在的图像目录；未下载时返回 None。"""
    for d in F30K_IMG_DIRS:
        if os.path.isdir(d) and any(n.endswith(".jpg") for n in os.listdir(d)[:50]):
            return d
    return None


def _spatial_edges(boxes_px: np.ndarray, iou_thr: float = 0.20) -> list[tuple]:
    """同一图像内的区域两两关系（确定性，只用像素坐标）。

    规则（都在**有序对**上判定，故方向唯一）：
      · 按中心横坐标排序后取**相邻**对 → `left_of`；
      · 按中心纵坐标排序后取**相邻**对 → `above`（图像坐标系 y 向下，故 cy 小者在上）；
      · 交并比 > `iou_thr` 的对 → `overlaps`（取覆盖关系更强的方向：面积小者在被包含侧）。
    只取"相邻对"是为了让稀疏度可控（每图 O(k) 条，而非 O(k²)），
    这与 4.2.2 的"结构画像取局部邻域"口径一致。
    """
    n = len(boxes_px)
    if n < 2:
        return []
    cx = (boxes_px[:, 0] + boxes_px[:, 2]) / 2.0
    cy = (boxes_px[:, 1] + boxes_px[:, 3]) / 2.0
    area = np.maximum((boxes_px[:, 2] - boxes_px[:, 0])
                      * (boxes_px[:, 3] - boxes_px[:, 1]), 0.0)
    out: list[tuple] = []
    for order, rel in ((np.argsort(cx, kind="stable"), "left_of"),
                       (np.argsort(cy, kind="stable"), "above")):
        for a, b in zip(order[:-1], order[1:]):
            if a != b:
                out.append((int(a), rel, int(b)))
    # 交并比：只保留中心点落在对方框内的方向，避免 O(k²) 全对枚举的噪声
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            x1 = max(boxes_px[a, 0], boxes_px[b, 0]); y1 = max(boxes_px[a, 1], boxes_px[b, 1])
            x2 = min(boxes_px[a, 2], boxes_px[b, 2]); y2 = min(boxes_px[a, 3], boxes_px[b, 3])
            inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
            if inter <= 0.0:
                continue
            iou = inter / max(area[a] + area[b] - inter, 1e-6)
            if iou > iou_thr and area[a] <= area[b]:
                out.append((a, "overlaps", b))
    return out


def load_flickr_regions(max_images: int | None = None, seed: int = 2026,
                        seed_ratio: float = 0.30, val_ratio_of_seeds: float = 0.10,
                        max_test: int | None = None) -> Pair:
    """把 Flickr30K Entities 构造成"视觉区域 ↔ 文本短语"对齐任务（Track A′）。

    构造（完全由真实标注决定，不引入任何人工标签）：
      · 左（视觉侧）：每个核心指代链的一张边界框；语义视图由**可训练的区域编码器**
        产出（`cussm.mm_encoder` 编码该框的像素裁剪，无编码器时退回几何构型）；
      · 右（文本侧）：每个核心指代链的一条短语；语义视图由**可训练的短语编码器**
        产出（无编码器时退回字符 n-gram TF-IDF）；
      · 对齐真值：两侧核心指代链 ID 相同的单元即互为对齐（`[/EN#id/type/phrase]`）。

    **2026-09-28 改造（与 5.1.1 列出的三条"不可评估"理由一一对应）**：

    ① 左侧区域图**补上关系边**。原实现的 `rel_triples` 是空数组，结构画像恒为空。
       现在按 `_spatial_edges` 由边界框几何确定性地导出 `left_of`／`above`／
       `overlaps` 三类关系 ⇒ 结构路由 $H$ 在该轨道上有支撑对象。
    ② 两侧特征改由**同一嵌入空间**产出（`cussm.mm_encoder` 的双塔），
       不再一边几何一边字面。
    ③ **堵住"真值恒等映射"这条泄漏通道**，三处同时改：
       · **按图像切分**——原实现按对切分，同一图像的其它链仍留在训练集，
         而真值只取决于"哪条链"，同图不构成泄漏；真正的风险是**同图区域的空间关系
         在两侧被同一套构造复现**，故改为 30% 图像作种子/验证、70% 图像作测试，
         两组图像**不交**；
       · **右侧下标随机置换**——原实现两侧按下标一一对齐（left[i]↔right[i]），
         任何能预测"自己排第几"的特征都会白拿分；置换后该通道关闭；
       · **左侧 surface 去标识**——原实现 `lsurf = k.split("#")[1]` 直接把**链号**
         写进左侧表层名，等于把答案放进特征；改为中性 id。

    本函数**不使用图像像素**（像素由 `cussm.mm_encoder` 另行读取），
    因此在不装编码器时它仍然只是"结构代理"，装了才是 Track A′。
    """
    zp = os.path.join(RAW, "flickr30k_entities", "annotations.zip")
    if not os.path.exists(zp):
        raise FileNotFoundError(f"未找到 {zp}，先运行 code/download_data.py --group f30k")

    left_rows: list[tuple] = []      # (chain_key, 几何, 像素框, 图像 id, 句号, 句内序号)
    right_rows: list[tuple] = []     # (chain_key, 短语, 类型, 图像 id, 句号, 句内序号)
    left_triples: list[tuple] = []   # 左侧空间关系（局部累加，勿提为模块级）
    right_triples: list[tuple] = []  # 右侧次序/类型关系（同上）
    # 关系名 -> 整数 id：三元组第 2 列必须是 int32，不能存字符串
    LREL = {n: i for i, n in enumerate(LEFT_SPATIAL_RELS)}
    RREL = {n: i for i, n in enumerate(RIGHT_ORDER_RELS)}
    with zipfile.ZipFile(zp) as z:
        ann = {n.split("/")[-1][:-4]: n for n in z.namelist()
               if n.startswith("Annotations/") and n.endswith(".xml")}
        sen = {n.split("/")[-1][:-4]: n for n in z.namelist()
               if n.startswith("Sentences/") and n.endswith(".txt")}
        ids = sorted(set(ann) & set(sen))
        if max_images:
            ids = ids[:max_images]
        for img in ids:
            xml = z.read(ann[img]).decode("utf-8", "replace")
            W = float(_tag(xml, "width") or 500.0)
            H = float(_tag(xml, "height") or 500.0)
            boxes: dict[str, list[tuple]] = {}
            for m in re.finditer(
                    r"<object>(.*?)</object>", xml, re.S):
                blk = m.group(1)
                ids_ = re.findall(r"<name>([^<]+)</name>", blk)
                bb = re.search(
                    r"<xmin>(\d+)</xmin>.*?<ymin>(\d+)</ymin>.*?"
                    r"<xmax>(\d+)</xmax>.*?<ymax>(\d+)</ymax>", blk, re.S)
                if not bb:
                    continue
                x1, y1, x2, y2 = map(float, bb.groups())
                for cid in ids_:
                    boxes.setdefault(cid, []).append((x1, y1, x2, y2))
            txt = z.read(sen[img]).decode("utf-8", "replace")
            # 逐句扫描：记录每条链**首次出现**的句号与句内序号（用于右侧关系）
            phrases: dict[str, tuple] = {}
            for si, line in enumerate(txt.splitlines()):
                if not line.strip():
                    continue
                for pos, (cid, typ, phr) in enumerate(_ENT_RE.findall(line)):
                    phrases.setdefault(cid, (phr.strip(), typ, si, pos))
            if not phrases:
                continue
            local: list[int] = []                       # 本图在 left_rows 中的下标
            img_boxes: list[tuple] = []
            for rank, (cid, bxs) in enumerate(sorted(boxes.items())):
                if cid not in phrases:
                    continue
                x1 = np.mean([b[0] for b in bxs]); y1 = np.mean([b[1] for b in bxs])
                x2 = np.mean([b[2] for b in bxs]); y2 = np.mean([b[3] for b in bxs])
                w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
                key = f"{img}#{cid}"
                phr, typ, si, pos = phrases[cid]
                local.append(len(left_rows))
                img_boxes.append((x1, y1, x2, y2))
                left_rows.append((key, np.array([
                    (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, w / W, h / H,
                    w * h / (W * H), w / h, len(bxs) / 5.0, rank / 20.0,
                ], dtype=np.float32), (x1, y1, x2, y2), img, si, pos))
                right_rows.append((key, phr, typ, img, si, pos))
            # ---- 左侧关系：同图内由几何导出（不读真值）----
            if len(img_boxes) >= 2:
                arr = np.asarray(img_boxes, dtype=np.float64)
                for a, rel, b in _spatial_edges(arr):
                    left_triples.append((local[a], LREL[rel], local[b]))

    # ---- 右侧关系：同图同句内由次序与类型导出（不读真值）----
    by_sent: dict[tuple, list[int]] = {}
    for i, r in enumerate(right_rows):
        by_sent.setdefault((r[3], r[4]), []).append(i)
    for _k, idxs in by_sent.items():
        idxs.sort(key=lambda i: right_rows[i][5])
        for a, b in zip(idxs[:-1], idxs[1:]):
            right_triples.append((a, RREL["precedes"], b))
        by_type: dict[str, list[int]] = {}
        for i in idxs:
            by_type.setdefault(right_rows[i][2], []).append(i)
        for _t, gi in by_type.items():
            for a, b in zip(gi[:-1], gi[1:]):
                right_triples.append((a, RREL["same_type"], b))

    lkeys = [r[0] for r in left_rows]
    rkeys = [r[0] for r in right_rows]
    if len(set(rkeys)) != len(rkeys):             # 保险：链 ID 全局唯一，不应触发
        seen, keep = set(), []
        for i, k in enumerate(rkeys):
            if k not in seen:
                seen.add(k)
                keep.append(i)
        right_rows = [right_rows[i] for i in keep]
        rkeys = [r[0] for r in right_rows]
    lmap = {k: i for i, k in enumerate(lkeys)}
    rmap = {k: i for i, k in enumerate(rkeys)}

    lp = np.stack([r[1] for r in left_rows]).astype(np.float32)
    # 右侧**不构造几何量**：其 modality 为 "textual"，特征由 attributes 块
    # （att 键＝类型标签、值＝短语）与表层名块（短语字面）构造，几何量不参与。
    # 原实现在此处写了 `rp = np.stack([r[1] for r in right_rows]).astype(np.float32)`，
    # 而 `right_rows` 的第 1 项是**短语字符串**，故必然抛
    # `ValueError: could not convert string to float` —— 且该值随后只赋给
    # `right.geom`（文本模态下从不被读取），属死代码。已删除，不再计算。
    # 左侧 surface 改为**中性 id**：原先直接写入链号 `k.split("#")[1]`，等于把真值
    # 放进特征（本轮堵漏，见 docstring ③）。
    lsurf = [f"region_{i}" for i in range(len(lkeys))]
    rsurf = [r[1] for r in right_rows]
    rtype = [r[2] for r in right_rows]
    lbox = np.asarray([r[2] for r in left_rows], dtype=np.float32).reshape(-1, 4)
    limg = [r[3] for r in left_rows]
    lsent = np.asarray([r[4] for r in left_rows], dtype=np.int32)

    # ---- 真值对的索引：先按链键对齐，再**随机置换右侧下标**（堵漏 ③）----
    # 注意置换的方向：`new[j] = old[rperm[j]]`，故键 k（旧下标 o）的新下标是
    # **逆置换** `inv[o]`，而不是 `rperm[o]` —— 写成后者会把真值对错配（实测：
    # 真值两端落到不同图像上，按图像切分随即报错）。
    rng = np.random.default_rng(seed)
    rperm = rng.permutation(len(rkeys))
    inv = np.empty_like(rperm)
    inv[rperm] = np.arange(len(rperm))
    rmap_p = {k: int(inv[rmap[k]]) for k in rmap}
    right_rows = [right_rows[int(i)] for i in rperm]
    rkeys = [rkeys[int(i)] for i in rperm]
    rsurf = [rsurf[int(i)] for i in rperm]
    rtype = [rtype[int(i)] for i in rperm]
    rimg = [right_rows[i][3] for i in range(len(rkeys))]
    rsent = np.asarray([right_rows[i][4] for i in range(len(rkeys))], dtype=np.int32)

    pairs_all = np.array([(lmap[k], rmap_p[k]) for k in lkeys], dtype=np.int32)
    # ---- **按图像切分**（堵漏 ③）：同一条链的左右两侧必然同图，故按图切分
    #      不会切断任何真值对，只保证训练/验证与测试的**图像集合不交** ----
    imgs = sorted(set(limg))
    ip = rng.permutation(len(imgs))
    n_img_seed = max(1, int(round(len(imgs) * seed_ratio)))
    seed_imgs = {imgs[i] for i in ip[:n_img_seed]}
    is_seed = np.array([im in seed_imgs for im in limg], dtype=bool)
    pairs_seed = pairs_all[is_seed]
    test = pairs_all[~is_seed]
    n_val = max(1, int(round(len(pairs_seed) * val_ratio_of_seeds)))
    vp = rng.permutation(len(pairs_seed))
    val, seeds = pairs_seed[vp[:n_val]], pairs_seed[vp[n_val:]]

    # ---- 关系三元组下标：左侧关系在置换前后不变；右侧关系需跟随置换重编号 ----
    r_new = {int(old): int(new) for old, new in enumerate(rperm)}
    rel_l = np.asarray(left_triples, dtype=np.int32).reshape(-1, 3)
    rel_r_raw = np.asarray(right_triples, dtype=np.int32).reshape(-1, 3)
    if len(rel_r_raw):
        rel_r = rel_r_raw.copy()
        rel_r[:, 0] = [r_new[int(i)] for i in rel_r_raw[:, 0]]
        rel_r[:, 2] = [r_new[int(i)] for i in rel_r_raw[:, 2]]
    else:
        rel_r = np.zeros((0, 3), np.int32)

    left = KG(name="flickr:region", ents=lkeys,
              rel_triples=rel_l,
              att_triples=[(i, "geom", "", None) for i in range(len(lkeys))],
              rel_names=list(LEFT_SPATIAL_RELS), e2i=lmap, surface=lsurf)
    # 视觉侧几何特征挂在 KG 上，供 features 层读取（不写进 att_triples 以免污染语义路由）
    left.geom = lp                                              # type: ignore[attr-defined]
    left.modality = "visual"                                    # type: ignore[attr-defined]
    left.boxes_px = lbox                                        # type: ignore[attr-defined]
    left.image_ids = limg                                       # type: ignore[attr-defined]
    left.sent_idx = lsent                                       # type: ignore[attr-defined]
    right = KG(name="flickr:phrase", ents=rkeys,
               rel_triples=rel_r,
               att_triples=[(i, rtype[i], rsurf[i], None) for i in range(len(rkeys))],
               rel_names=list(RIGHT_ORDER_RELS), e2i=rmap_p, surface=rsurf)
    right.modality = "textual"                                  # type: ignore[attr-defined]
    right.types = rtype                                         # type: ignore[attr-defined]
    right.image_ids = rimg                                      # type: ignore[attr-defined]
    right.sent_idx = rsent                                      # type: ignore[attr-defined]

    n_img = len(set(limg))
    # `max_test` 只限制**参与评测的查询数**（加快调试），不改变候选集 ——
    # 与 DBP15K/SAST 加载器同一口径（见 `load_dbp15k` 的同名参数）。
    # Track A′ 的候选集有 19 万量级，全量打分需分块；抽样查询用于冒烟。
    if max_test and len(test) > max_test:
        test = test[np.sort(rng.choice(len(test), max_test, replace=False))]
    return Pair(name="Flickr30K-Entities", left=left, right=right,
                seeds=seeds, val=val, test=test,
                meta={"dataset": "Flickr30K-Entities", "unit": "区域↔短语",
                      "truth_per_left": 1, "candidates": f"全部 {len(rkeys)} 条文本短语",
                      "images": n_img, "n_chain": len(lkeys),
                      "types": sorted(set(rtype)),
                      "img_split": f"按图像切分：种子/验证图像 {n_img_seed} 张、测试图像 "
                                   f"{len(imgs) - n_img_seed} 张（不交）",
                      "right_permuted": True,
                      "left_rels": f"{len(rel_l)} 条（{LEFT_SPATIAL_RELS}）",
                      "right_rels": f"{len(rel_r)} 条（{RIGHT_ORDER_RELS}）",
                      "image_dir": flickr_image_dir(),
                      "note": "Track A′：像素由 cussm.mm_encoder 读取，本加载器不含像素"})


def _tag(xml: str, name: str):
    m = re.search(rf"<{name}>([^<]+)</{name}>", xml)
    return m.group(1) if m else None


# ================================================================ 3. 小规模 KG 自检
def load_countries_s1(seed: int = 2026, seed_ratio: float = 0.30,
                      max_test: int | None = None):
    """把 countries_S1 构造成受控对齐任务：右图 = 左图全量；左图 = 去掉一半关系边的同图。

    用途是**单元自检与受控结构实验**（结构信号强度可调），不是公开基准。
    """
    d = os.path.join(RAW, "kg_small", "data", "countries_S1", "train.txt")
    if not os.path.exists(d):
        raise FileNotFoundError(f"未找到 {d}，先运行 code/download_data.py --group small")
    raw = []
    with open(d, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) == 3:
                raw.append(p)
    ents = sorted({u for h, r, t in raw for u in (h, t)})
    e2i = {u: i for i, u in enumerate(ents)}
    # 关系键必须用**确定性**映射：Python 内建 hash() 受 PYTHONHASHSEED 影响，不可用于复现实验
    rel_names: list[str] = []
    r2i: dict[str, int] = {}
    for _, r, _ in raw:
        if r not in r2i:
            r2i[r] = len(rel_names)
            rel_names.append(r)
    tri = np.array([[e2i[h], r2i[r], e2i[t]] for h, r, t in raw], dtype=np.int32)
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(tri), dtype=bool)
    keep[rng.permutation(len(tri))[: int(len(tri) * 0.5)]] = True
    left = KG("countries:sparse", ents, tri[keep], [], rel_names, e2i)
    right = KG("countries:full", ents, tri, [], rel_names, e2i)
    pairs = np.array([(i, i) for i in range(len(ents))], dtype=np.int32)
    perm = rng.permutation(len(pairs))
    n_seed = int(round(len(pairs) * seed_ratio))
    test_pairs = pairs[perm[:n_seed]]
    # `max_test` 只限制**参与评测的查询数**，不改变候选集 —— 与 dbp15k / sast / flickr 同口径。
    # 2026-09-28 补：此前本加载器缺该参数，导致 `run_experiment.py --max-test` 在此数据集上
    # 抛 TypeError（`got an unexpected keyword argument 'max_test'`）并整体失败。
    if max_test and len(test_pairs) > max_test:
        test_pairs = test_pairs[np.sort(rng.choice(len(test_pairs), max_test,
                                                   replace=False))]
    return Pair("countries-S1-受控", left, right, pairs[perm[n_seed:]],
                pairs[perm[: max(1, n_seed // 10)]], test_pairs,
                {"dataset": "countries_S1", "unit": "实体(受控)", "truth_per_left": 1,
                 "note": "右图为左图补全边的同图，用于自检；非公开基准"})


# ================================================================ 4. 中等规模标准基准 → SAST
# 数据集登记：data/raw 下的相对目录 → 显示名。目录布局为
#   data/raw/<ds_key>/data/<官方目录名>/{train,valid,test}.txt
_PLAIN_DS = {
    "fb15k237": ("fb15k237/data/FB15k-237", "FB15k-237"),
    "wn18": ("wn18/data/wn18", "WN18"),
    "yago3_10": ("yago3_10/data/YAGO3-10", "YAGO3-10"),
}


def _opaque(side: str, i: int) -> str:
    """给实体一个**不可反推**的假表层名（用于屏蔽字面信息）。

    必须满足两条：① 两侧同一下标得到不同字符串（否则名字路由白送答案）；
    ② 同侧不同实体之间没有可共享的字符 n-gram（否则名字路由会按"数字相近"聚类）。
    故用 blake2b 摘要再 base32 编码，取 13 个字符 —— 长度固定、字符集固定、
    但内容与下标无任何单调关系。
    """
    import base64
    import hashlib
    h = hashlib.blake2b(f"{side}:{i}".encode("utf-8"), digest_size=8).digest()
    return "z" + base64.b32encode(h).decode("ascii").rstrip("=")


def _load_plain_kg(dirpath: str, disp: str, shuffle_seed: int = 0):
    """读 DeepGraphLearning 仓库格式的 KG。

    文件格式实测（2026-09-27）：`train.txt/valid.txt/test.txt` 为**制表符**分隔、
    CRLF 行尾、字段是实体/关系的**名称**（不是 dict 里的整数 id）；
    `entities.dict` / `relations.dict` 为 `整数id \\t 名称`。

    返回 (实体名列表, 关系名列表, 三元组 int32 数组)。
    """
    d = os.path.join(RAW, *dirpath.split("/"))
    for fn in ("train.txt", "valid.txt", "test.txt"):
        if not os.path.exists(os.path.join(d, fn)):
            raise FileNotFoundError(
                f"未找到 {os.path.join(d, fn)}；先运行 "
                f"code/download_data.py --group kgbench")

    def read_dict(fn):
        out = []
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            return out
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                q = line.rstrip("\r\n").split("\t")
                if len(q) >= 2:
                    out.append(q[1].strip())
        return out

    ent_list = read_dict("entities.dict")
    if not ent_list:
        raise FileNotFoundError(f"{d}/entities.dict 缺失或为空")
    rel_list = read_dict("relations.dict")
    e2i = {e: i for i, e in enumerate(ent_list)}
    r2i: dict[str, int] = {r: i for i, r in enumerate(rel_list)}

    raw = []
    for fn in ("train.txt", "valid.txt", "test.txt"):
        with open(os.path.join(d, fn), encoding="utf-8", errors="replace") as f:
            for line in f:
                q = line.rstrip("\r\n").split("\t")
                if len(q) != 3:
                    q2 = line.split()
                    if len(q2) != 3:
                        continue
                    q = q2
                raw.append((q[0].strip(), q[1].strip(), q[2].strip()))
    for _h, r, _t in raw:
        if r not in r2i:
            r2i[r] = len(r2i)

    rnames = [""] * len(r2i)
    for r, i in r2i.items():
        rnames[i] = r
    tri = np.array([[e2i[h], r2i[r], e2i[t]] for h, r, t in raw], dtype=np.int32)

    ents = list(ent_list)
    if shuffle_seed:
        perm = np.random.default_rng(shuffle_seed).permutation(len(ents))
        ents = [ents[i] for i in perm]
    return ents, rnames, tri


def load_sast(dataset: str = "fb15k237", split: float = 0.7, seed: int = 2026,
              mask_names: bool = True, seed_ratio: float = 0.30,
              val_ratio_of_seeds: float = 0.10, max_test: int | None = None) -> Pair:
    """SAST（Self-Alignment Stress Test，自对齐压力测试）。

    **为什么要这个任务**：FB15k-237 / WN18 / YAGO3-10 是公认的 KG 嵌入基准，
    但它们只有**一张图**，本身不含跨图对齐真值，无法直接产出 Hits@1。本函数把
    单张图按边随机切成两张"观测"，从而**人造出一个真值完全已知**的匹配任务：

      · 把同一张图的边随机拆成两份（默认 70% / 30%），左侧只看到 70% 的边，
        右侧只看到 30% 的边；两侧共享同一套关系词表；
      · 左侧实体 = 在左视图中出现过的实体；右侧同理；两侧**各自独立的、并被打乱
        顺序的下标空间**（切断下标侧信道）；
      · 对齐真值 = 同一个全局实体在两侧的下标对 —— **真值精确已知，不依赖任何标注**；
      · 若 `mask_names=True`，两侧实体的表层名被替换为不可反推的假名，
        于是"语义路由"只能依赖**关系名画像**（实体参与的关系名多重集），
        这正是一个实体在 KG 中的"类型语义"，且两侧词表共享、真实可比。

    **它测的是什么**：两视图是同一结构的两份稀疏观测，一个保结构的匹配算子
    应当能把同一实体的两次观测对应起来。因此该指标直接度量"结构保持能力"，
    且**没有名字泄露**，是纯结构与关系语义的对照实验。它不是公开基准的替代品，
    在论文中作为 Track D，与 Track B（DBP15K 跨语言）并列。
    """
    if dataset not in _PLAIN_DS:
        raise KeyError(f"未知数据集 {dataset}，可选 {sorted(_PLAIN_DS)}")
    rel_dir, disp = _PLAIN_DS[dataset]

    ents, rnames, tri = _load_plain_kg(rel_dir, disp)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tri))
    n_l = int(round(len(tri) * split))
    tri_l, tri_r = tri[perm[:n_l]], tri[perm[n_l:]]

    def side_kg(T, side):
        used = np.unique(np.concatenate([T[:, 0], T[:, 2]])) if len(T) else np.zeros(0, np.int64)
        order = rng.permutation(len(used)) if side == "R" else np.arange(len(used))
        glob = used[order]
        loc = {int(g): k for k, g in enumerate(glob)}
        T2 = np.empty((len(T), 3), np.int32)
        T2[:, 0] = [loc[int(x)] for x in T[:, 0]]
        T2[:, 1] = T[:, 1]
        T2[:, 2] = [loc[int(x)] for x in T[:, 2]]
        surf = [_opaque(side, int(g)) for g in glob] if mask_names \
            else [local_name(ents[int(g)]) for g in glob]
        # 关系名画像：每个实体参与的**关系名**多重集（去重后一条一个 att），
        # 作为该实体的语义内容 —— 两侧词表相同，故真实可比。
        prof: dict[int, set] = {}
        for a, r, b in T2:
            prof.setdefault(int(a), set()).add(int(r))
            prof.setdefault(int(b), set()).add(int(r))
        att = []
        for e, rs in prof.items():
            for r in sorted(rs):
                att.append((e, "rel", rnames[r], None))
        kg = KG(name=f"SAST-{disp}:{side}", ents=[str(g) for g in glob],
                rel_triples=T2, att_triples=att, rel_names=list(rnames),
                e2i={str(g): k for k, g in enumerate(glob)}, surface=surf)
        return kg, glob, loc

    left, gL, locL = side_kg(tri_l, "L")
    right, gR, locR = side_kg(tri_r, "R")

    common = sorted(set(gL.tolist()) & set(gR.tolist()))
    ill = np.array([[locL[g], locR[g]] for g in common], dtype=np.int32)

    p = rng.permutation(len(ill))
    n_seed = max(2, int(round(len(ill) * seed_ratio)))
    s_idx, t_idx = p[:n_seed], p[n_seed:]
    n_val = max(1, int(round(n_seed * val_ratio_of_seeds)))
    val, seeds = ill[s_idx[:n_val]], ill[s_idx[n_val:]]
    test = ill[t_idx]
    if max_test and len(test) > max_test:
        test = test[np.sort(rng.choice(len(test), max_test, replace=False))]

    return Pair(name=f"SAST-{disp}", left=left, right=right,
                seeds=seeds, val=val, test=test,
                meta={"dataset": disp, "task": "SAST（自对齐压力测试）",
                      "unit": "实体", "truth_per_left": 1, "seed": seed,
                      "split": split, "mask_names": mask_names,
                      "seed_ratio": seed_ratio,
                      "candidates": f"右侧全部 {right.n} 个实体",
                      "n_ill": int(len(ill)), "n_edges": int(len(tri)),
                      "note": "同一张图的边随机二分 → 两视图；真值=同一全局实体；"
                              "名称已屏蔽，语义路由只吃关系名画像"})


# ================================================================ 枚举
LOADERS = {
    "dbp15k_fr_en": lambda **kw: load_dbp15k("fr_en", **kw),
    "dbp15k_zh_en": lambda **kw: load_dbp15k("zh_en", **kw),
    "dbp15k_ja_en": lambda **kw: load_dbp15k("ja_en", **kw),
    "flickr30k_entities": load_flickr_regions,
    "countries_s1": load_countries_s1,
    "fb15k237": lambda **kw: load_sast("fb15k237", **kw),
    "wn18": lambda **kw: load_sast("wn18", **kw),
    "yago3_10": lambda **kw: load_sast("yago3_10", **kw),
}
