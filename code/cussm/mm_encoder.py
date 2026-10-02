# -*- coding: utf-8 -*-
r"""端到端跨模态编码器 —— Track A′ 的两塔与两阶段训练。

## 这个模块解决什么问题

第 5.1.1 节判定 Track A′ 「不可评估」，三条理由里的第二条是**两侧特征不共享空间**：
左侧只用边界框几何、右侧只用短语字面，二者之间不存在可比较的度量。本模块用一对
**可训练的编码器**把两侧映射到同一嵌入空间：

    · 视觉塔：ViT-B/16 编码**区域裁剪**（按归一化边界框从原图裁出，缩放到 224²）
    · 文本塔：BERT-base 编码短语

两塔各接一个投影头（768 → d），投影头输出即第 4 章 $\mathcal{R}_m$ 的"模态编码器"
产物，回填给 `features.build_view` 作为**语义视图**（见 `attach`）。

## 两阶段训练（对应 5.4.3 的口径）

* **阶段一：线性探测。** 冻结两塔骨干，**只缓存骨干输出**（768 维），在此之上训
  投影头。关键点：缓存的是**骨干**而不是投影后的向量 —— 若缓存投影输出再回头训
  投影头，前后就不一致（初始头的输出被当作不变特征），这是本轮实现时踩到并修掉的
  第一个结构错误。
* **阶段二：端到端微调。** 解冻骨干，投影头用大学习率（`lr_head`）、骨干用小学习率
  （`lr_bb`），逐小批前向 + 反向（**不能**复用阶段一的缓存路径，那条路径带
  `torch.no_grad`）。

两阶段的损失都是**对称 InfoNCE**（温度 $\tau$）：

    L = -½ [ CE(z_L z_R^T / τ, 对角) + CE(z_R z_L^T / τ, 对角) ]

选 InfoNCE 而非 MSE 的理由与 `torch_backend.train_joint` 的"为什么对齐损失必须带
负采样"同源：纯距离目标存在**平凡极小**（把两侧都收缩到同一点即可），负例把塌缩
挡在外面。

## 权重来源与可达性

`huggingface.co` 在本项目的云端实例上**不可达**（2026-09-28 实测超时），
`hf-mirror.com` 通达。故本模块在导入 transformers **之前**把 `HF_ENDPOINT`
指向镜像（见 `set_hf_mirror`）。这是环境事实，不是偏好。

## 与"同一份代码路径"纪律的关系

本模块不改变任何既有方法的打分逻辑：它只**替换** Track A′ 两侧的语义视图来源。
CPU/GPU 的切换沿用 `cussm.dev` 的设备解析（`--device` / `CUSSM_DEVICE`）。
"""
from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass, field

import numpy as np

from . import data as D

# ----------------------------------------------------------------- 镜像与依赖


def set_hf_mirror() -> str:
    """把 HF 端点指到国内镜像（必须在 `import transformers` 之前调用）。"""
    ep = os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"
    os.environ["HF_ENDPOINT"] = ep
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return ep


def has_deps() -> bool:
    """两塔所需的依赖是否齐备（transformers + Pillow + torch）。"""
    try:
        import PIL  # noqa: F401
        import torch  # noqa: F401
        set_hf_mirror()
        import transformers  # noqa: F401
        return True
    except Exception:                                    # noqa: BLE001
        return False


def deps_report() -> dict:
    out = {"torch": None, "transformers": None, "pillow": None, "hf_endpoint": None}
    try:
        import torch
        out["torch"] = torch.__version__
    except Exception:                                    # noqa: BLE001
        pass
    try:
        import PIL
        out["pillow"] = getattr(PIL, "__version__", "?")
    except Exception:                                    # noqa: BLE001
        pass
    try:
        set_hf_mirror()
        import transformers
        out["transformers"] = transformers.__version__
        out["hf_endpoint"] = os.environ.get("HF_ENDPOINT")
    except Exception:                                    # noqa: BLE001
        pass
    return out


# ----------------------------------------------------------------- 图像访问

F30K_ZIP = os.path.join(D.RAW, "flickr30k", "flickr30k-images.zip")


class ImageBank:
    """Flickr30K 图像的随机访问（zip 或目录）。

    zip 走 `ZipFile.read(member)` 的按需读，**不必解压** —— 解压后的 31,783 张
    JPEG 会白占约 4.4 GB 磁盘，而实例的系统盘只有 50 GB。

    **两处加速（2026-09-28 实测必需）**：原实现对**每个区域**都
    ① `zipfile.ZipFile(...)` 重开一次、② 把该区域所在图像的 JPEG 重新解码一遍。
    Track A′ 全量要取 193,003 个区域裁剪，而每张图平均只有 ~6 个区域
    ⇒ ① 让每次读取都重扫 4.1 GB 归档的中央目录、② 让 JPEG 被重复解码 ~6 倍，
    实测吞吐 **7.1 区域/s**（193k 区域 ≈ 7.5 h，只算阶段一的取图）。
    改为「进程内常驻句柄 + 上一张已解码图像（LRU-1）」后按区域下标的**有序遍历**
    （`RegionCropDataset` 正是有序的，同图区域连续）几乎全部命中缓存。

    并发约定：句柄与缓存都是**进程内**状态。DataLoader 的 worker 是 fork 出来的
    独立进程，各自持有自己的句柄（按 `os.getpid()` 判定，fork 后自动重开），
    因此不存在线程/进程间共享问题；但**不要**把同一个 `ImageBank` 实例交给多线程使用。
    """

    def __init__(self):
        d = D.flickr_image_dir()
        self.dir_path: str | None = None
        self.zip_path: str | None = None
        self._members: dict = {}
        self._zip = None            # 常驻归档句柄（按 pid 绑定，见 _handle）
        self._zip_pid: int | None = None
        self._cache_id: str | None = None
        self._cache_im = None
        if d:
            self.dir_path = d
            return
        if os.path.exists(F30K_ZIP):
            self.zip_path = F30K_ZIP
            with zipfile.ZipFile(F30K_ZIP) as z:
                for n in z.namelist():
                    if n.lower().endswith((".jpg", ".jpeg")):
                        self._members[os.path.basename(n)[:-4]] = n
            return
        raise FileNotFoundError(
            f"未找到 Flickr30K 图像：{F30K_ZIP} 或 {D.F30K_IMG_DIRS} 均不存在。")

    def _handle(self):
        """取本进程的常驻 ZipFile；fork 出子进程后 pid 变化会自动重开。"""
        pid = os.getpid()
        if self._zip is None or self._zip_pid != pid:
            try:
                if self._zip is not None:
                    self._zip.close()
            except Exception:                            # noqa: BLE001
                pass
            self._zip = zipfile.ZipFile(self.zip_path)
            self._zip_pid = pid
            self._cache_id, self._cache_im = None, None   # 跨进程缓存作废
        return self._zip

    def n_images(self) -> int:
        if self.dir_path:
            return sum(1 for n in os.listdir(self.dir_path)
                       if n.lower().endswith((".jpg", ".jpeg")))
        return len(self._members)

    def read(self, img_id: str):
        """返回 PIL.Image（RGB）；img_id 为不带扩展名的文件名。

        命中 LRU-1 时返回**同一个对象**（不复制）：调用方只做 `.size` 读取与
        `.crop()`（返回新对象，不改本体），不得就地修改返回值。
        """
        from PIL import Image
        if self.dir_path:
            for ext in (".jpg", ".jpeg", ".png"):
                p = os.path.join(self.dir_path, img_id + ext)
                if os.path.exists(p):
                    return Image.open(p).convert("RGB")
            raise KeyError(f"目录中没有图像 {img_id}")
        if self._zip_pid != os.getpid():
            self._handle()                               # fork 后先重建句柄与缓存
        if self._cache_id == img_id and self._cache_im is not None:
            return self._cache_im
        member = self._members.get(img_id)
        if member is None:
            raise KeyError(f"zip 中没有图像 {img_id}")
        blob = self._handle().read(member)
        im = Image.open(io.BytesIO(blob)).convert("RGB")
        self._cache_id, self._cache_im = img_id, im
        return im


def normalized_boxes(pair) -> np.ndarray:
    """从 `pair.left.geom` 反解**归一化**边界框 (N,4)，取值 0–1。

    这样做回避了"标注里的 width/height 与 JPEG 实际尺寸可能不一致"这一坑：
    `geom` 第 0–3 列已是除以标注宽高的中心与边长，反解出的框天然落在 0–1，
    再乘上**实际**图像尺寸即得像素框，不需要知道标注的宽高。
    """
    g = np.asarray(pair.left.geom, dtype=np.float64)
    cx, cy, w, h = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
    return np.stack([np.clip(cx - w / 2.0, 0.0, 1.0),
                     np.clip(cy - h / 2.0, 0.0, 1.0),
                     np.clip(cx + w / 2.0, 0.0, 1.0),
                     np.clip(cy + h / 2.0, 0.0, 1.0)], axis=1)


def crop_one(bank: ImageBank, img_id: str, box, size: int) -> np.ndarray:
    from PIL import Image
    try:
        im = bank.read(img_id)
    except Exception:                                    # noqa: BLE001
        return np.zeros((size, size, 3), np.uint8)
    W, H = im.size
    x1, y1, x2, y2 = box
    px1, py1 = int(x1 * W), int(y1 * H)
    px2 = max(int(x2 * W), px1 + 2)
    py2 = max(int(y2 * H), py1 + 2)
    crop = im.crop((px1, py1, px2, py2)).resize((size, size), Image.BICUBIC)
    return np.asarray(crop, dtype=np.uint8)


def load_crops(pair, bank: ImageBank, idx, size: int = 224) -> np.ndarray:
    """把指定区域下标批量裁成 (n, size, size, 3) uint8。"""
    boxes = normalized_boxes(pair)
    imgs = pair.left.image_ids
    out = np.empty((len(idx), size, size, 3), np.uint8)
    for k, i in enumerate(np.asarray(idx)):
        out[k] = crop_one(bank, imgs[int(i)], boxes[int(i)], size)
    return out


class RegionCropDataset:
    """`load_crops` 的 Dataset 包装，交给 DataLoader 并行取图。"""

    def __init__(self, pair, bank: ImageBank, size: int = 224, idx=None):
        self.pair, self.bank, self.size = pair, bank, size
        self.boxes = normalized_boxes(pair)
        self.imgs = pair.left.image_ids
        self.idx = np.arange(pair.left.n) if idx is None else np.asarray(idx)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int):
        i = int(self.idx[k])
        return crop_one(self.bank, self.imgs[i], self.boxes[i], self.size)


# ----------------------------------------------------------------- 两塔


class DualTower:
    """ViT-B/16（区域）+ BERT-base（短语）→ 共用 d 维空间。

    公开方法只暴露"算骨干特征"与"用投影头出嵌入"两步，因为两阶段训练需要
    **分开复用**它们：阶段一缓存骨干特征、只训头；阶段二两者都带梯度。
    """

    def __init__(self, d: int = 256, vis: str = "google/vit-base-patch16-224",
                 txt: str = "bert-base-uncased", device=None, tau: float = 0.07,
                 max_len: int = 24):
        set_hf_mirror()
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
        self.torch = torch
        self.d, self.tau, self.max_len = d, tau, max_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vis_name, self.txt_name = vis, txt
        self.vis_proc = AutoImageProcessor.from_pretrained(vis)
        self.tok = AutoTokenizer.from_pretrained(txt)
        self.vis_bb = AutoModel.from_pretrained(vis).to(self.device)
        self.txt_bb = AutoModel.from_pretrained(txt).to(self.device)
        self.d_v = int(self.vis_bb.config.hidden_size)
        self.d_t = int(self.txt_bb.config.hidden_size)
        self.head_v = torch.nn.Linear(self.d_v, d).to(self.device)
        self.head_t = torch.nn.Linear(self.d_t, d).to(self.device)

    # ---- 骨干特征（两阶段都要用）----
    def feat_regions(self, crops_u8) -> "torch.Tensor":
        proc = self.vis_proc(images=[a for a in crops_u8], return_tensors="pt")
        px = proc["pixel_values"].to(self.device)
        return self.vis_bb(pixel_values=px).last_hidden_state[:, 0]

    def feat_phrases(self, texts) -> "torch.Tensor":
        enc = self.tok(list(texts), padding=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        out = self.txt_bb(input_ids=enc["input_ids"].to(self.device),
                          attention_mask=enc["attention_mask"].to(self.device))
        return out.last_hidden_state[:, 0]

    # ---- 嵌入（投影 + L2）----
    @staticmethod
    def _proj(z, head):
        import torch.nn.functional as F
        return F.normalize(head(z), dim=-1)

    def regions(self, crops_u8) -> "torch.Tensor":
        return self._proj(self.feat_regions(crops_u8), self.head_v)

    def phrases(self, texts) -> "torch.Tensor":
        return self._proj(self.feat_phrases(texts), self.head_t)

    # ---- 损失 ----
    def infonce(self, zl, zr):
        t = self.torch
        logits = (zl @ zr.t()) / self.tau
        lab = t.arange(zl.shape[0], device=zl.device)
        return 0.5 * (t.nn.functional.cross_entropy(logits, lab)
                      + t.nn.functional.cross_entropy(logits.t(), lab))

    # ---- 可训练性与优化器 ----
    def set_trainable(self, backbone: bool) -> None:
        for m in (self.vis_bb, self.txt_bb):
            for p in m.parameters():
                p.requires_grad_(backbone)

    def optimizer(self, backbone: bool, lr_head: float, lr_bb: float):
        t = self.torch
        groups = [{"params": list(self.head_v.parameters())
                           + list(self.head_t.parameters()), "lr": lr_head}]
        if backbone:
            groups.append({"params": list(self.vis_bb.parameters())
                                   + list(self.txt_bb.parameters()), "lr": lr_bb})
        return t.optim.AdamW(groups)


# ----------------------------------------------------------------- 批量前向


def embed_regions_raw(tower: DualTower, ds, batch: int = 64,
                      workers: int = 4, log_every: int = 0) -> np.ndarray:
    """缓存**骨干特征**（不是投影后的嵌入）—— 阶段一训投影头的前提。"""
    import torch
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                    drop_last=False)
    out, seen = [], 0
    tower.vis_bb.eval()
    with torch.no_grad():
        for arr in dl:
            out.append(tower.feat_regions(arr.numpy()).float().cpu().numpy())
            seen += len(arr)
            if log_every and seen % (log_every * batch) < batch:
                print(f"    区域骨干特征 {seen}/{len(ds)}", flush=True)
    return (np.concatenate(out, 0).astype(np.float32) if out
            else np.zeros((0, tower.d_v), np.float32))


def embed_phrases_raw(tower: DualTower, texts, batch: int = 256,
                      log_every: int = 0) -> np.ndarray:
    import torch
    out = []
    tower.txt_bb.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            out.append(tower.feat_phrases(texts[i:i + batch]).float().cpu().numpy())
            if log_every and (i // batch + 1) % log_every == 0:
                print(f"    短语骨干特征 {min(i+batch, len(texts))}/{len(texts)}", flush=True)
    return (np.concatenate(out, 0).astype(np.float32) if out
            else np.zeros((0, tower.d_t), np.float32))


def project_all(tower: DualTower, Fv: np.ndarray, Ft: np.ndarray,
                batch: int = 4096) -> tuple:
    """用当前投影头把骨干特征映射到 d 维并 L2 归一化（推理用，无梯度）。"""
    import torch
    tower.vis_bb.eval()
    outs = []
    with torch.no_grad():
        for name, F, head, dv in (("v", Fv, tower.head_v, tower.d_v),
                                  ("t", Ft, tower.head_t, tower.d_t)):
            acc = []
            for i in range(0, len(F), batch):
                x = torch.as_tensor(F[i:i + batch], device=tower.device)
                acc.append(DualTower._proj(x, head).float().cpu().numpy())
            outs.append(np.concatenate(acc, 0).astype(np.float32) if acc
                        else np.zeros((0, tower.d), np.float32))
    return outs[0], outs[1]


# ----------------------------------------------------------------- 两阶段训练


@dataclass
class TrainReport:
    stage1: dict = field(default_factory=dict)
    stage2: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict:
        return {"stage1": self.stage1, "stage2": self.stage2,
                "env": self.env, "note": self.note}


def train_two_stage(pair, epochs1: int = 6, epochs2: int = 1, d: int = 256,
                    bs: int = 64, bs2: int = 16, workers: int = 4,
                    lr_head: float = 1e-3, lr_bb: float = 1e-5,
                    seed: int = 2026, device=None, bank: ImageBank | None = None,
                    limit_regions: int | None = None,
                    stage2_pairs: int | None = 20000,
                    verbose: bool = True) -> tuple:
    """两阶段训练，返回 `(zL, zR, TrainReport)`。

    监督**只用** `pair.seeds`；**从不读 `pair.test`**（`val` 仅用于选择阶段二的轮数）。
    `limit_regions` 用于在冒烟时缩小规模；未编码的区域以零向量占位，并在报告里明写
    占位数量 —— 冒烟读数不得当成完整读数。
    """
    import torch
    torch.manual_seed(seed)
    bank = bank or ImageBank()
    tower = DualTower(d=d, device=device)
    nL, nR = pair.left.n, pair.right.n
    rng = np.random.default_rng(seed)
    rep = TrainReport(env=deps_report())
    rep.env["resolved_device"] = str(tower.device)

    # ---------------- 阶段一：冻结骨干，只训投影头 ----------------
    tower.set_trainable(backbone=False)
    sub = (np.sort(rng.choice(nL, limit_regions, replace=False))
           if limit_regions and limit_regions < nL else np.arange(nL))
    ds = RegionCropDataset(pair, bank, idx=sub)
    if verbose:
        print(f"  [阶段一] 取区域裁剪 {len(ds)} 个（共 {nL}）…", flush=True)
    Fv = np.zeros((nL, tower.d_v), np.float32)
    Fv[ds.idx] = embed_regions_raw(tower, ds, batch=bs, workers=workers, log_every=10)
    Ft = embed_phrases_raw(tower, pair.right.surfaces(), batch=256, log_every=20)

    tr = torch.as_tensor(pair.seeds, dtype=torch.long, device=tower.device)
    Fv_t = torch.as_tensor(Fv, device=tower.device)
    Ft_t = torch.as_tensor(Ft, device=tower.device)
    opt = tower.optimizer(backbone=False, lr_head=lr_head, lr_bb=lr_bb)
    hist1 = []
    for ep in range(epochs1):
        perm = rng.permutation(len(tr))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), 1024):
            sel = torch.as_tensor(perm[i:i + 1024], device=tower.device)
            li, ri = tr[sel, 0], tr[sel, 1]
            opt.zero_grad(set_to_none=True)
            loss = tower.infonce(DualTower._proj(Fv_t[li], tower.head_v),
                                 DualTower._proj(Ft_t[ri], tower.head_t))
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        hist1.append(tot / max(nb, 1))
        if verbose:
            print(f"  [阶段一] epoch {ep+1}/{epochs1}  InfoNCE={hist1[-1]:.4f}", flush=True)
    rep.stage1 = {"epochs": epochs1, "loss": hist1, "n_seeds": int(len(pair.seeds)),
                  "regions_encoded": int(len(ds.idx)), "regions_total": int(nL),
                  "regions_placeholder": int(nL - len(ds.idx)),
                  "phrases_encoded": int(nR), "d_backbone_v": int(tower.d_v),
                  "d_backbone_t": int(tower.d_t), "d": int(d), "tau": float(tower.tau),
                  "lr_head": lr_head, "bs": bs}

    # ---------------- 阶段二：解冻骨干微调 ----------------
    if epochs2 > 0:
        tower.set_trainable(backbone=True)
        opt2 = tower.optimizer(backbone=True, lr_head=lr_head, lr_bb=lr_bb)
        pool = pair.seeds if len(pair.seeds) <= (stage2_pairs or 10 ** 9) else \
            pair.seeds[np.sort(rng.choice(len(pair.seeds), stage2_pairs, replace=False))]
        surf = pair.right.surfaces()
        hist2 = []
        for ep in range(epochs2):
            order = rng.permutation(len(pool))
            tot, nb = 0.0, 0
            for i in range(0, len(order), bs2):
                b = pool[order[i:i + bs2]]
                if len(b) < 2:
                    continue
                li = np.sort(np.unique(b[:, 0]))
                ri = np.sort(np.unique(b[:, 1]))
                crops = load_crops(pair, bank, li)
                zl = tower.regions(crops)
                zr = tower.phrases([surf[int(x)] for x in ri])
                pl = {int(v): k for k, v in enumerate(li)}
                pr = {int(v): k for k, v in enumerate(ri)}
                keep = [(pl[int(a)], pr[int(x)]) for a, x in b
                        if int(a) in pl and int(x) in pr]
                if len(keep) < 2:
                    continue
                zl2 = zl[[p for p, _ in keep]]
                zr2 = zr[[q for _, q in keep]]
                opt2.zero_grad(set_to_none=True)
                loss = tower.infonce(zl2, zr2)
                loss.backward()
                opt2.step()
                tot += float(loss.detach())
                nb += 1
            hist2.append(tot / max(nb, 1))
            if verbose:
                print(f"  [阶段二] epoch {ep+1}/{epochs2}  InfoNCE={hist2[-1]:.4f}"
                      f"  ({nb} 批)", flush=True)
        rep.stage2 = {"epochs": epochs2, "loss": hist2, "bs": bs2,
                      "lr_bb": lr_bb, "lr_head": lr_head,
                      "pairs_used": int(len(pool)), "n_batch": int(nb)}
        if verbose:
            print("  [阶段二] 重算全量嵌入 …", flush=True)
        Fv = embed_regions_raw(tower, RegionCropDataset(pair, bank),
                               batch=bs, workers=workers, log_every=10)
        Ft = embed_phrases_raw(tower, surf, batch=256, log_every=20)
    else:
        rep.stage2 = {"epochs": 0, "note": "未执行（epochs2=0），仅有线性探测结果"}

    zL, zR = project_all(tower, Fv, Ft)
    rep.note = (f"双塔 ViT-B/16 + BERT-base，d={d}，τ={tower.tau}；"
                f"监督只用 seeds，未读 test；设备 {tower.device}")
    return zL, zR, rep


# ----------------------------------------------------------------- 评测


def evaluate_retrieval(zL: np.ndarray, zR: np.ndarray, pairs: np.ndarray,
                       ks=(1, 5, 10), chunk: int = 512, device=None) -> dict:
    """双向检索的 R@K，分块计算（不物化 查询数×候选数 的完整矩阵）。

    并列裁决与全项目一致：**降序稳定排序 + 并列按下标小者**，故

        rank = #{严格大于真值} + #{相等且下标更小} + 1

    分块只为省内存：193k 候选下，512×193k 的 float32 块约 396 MB，
    而整块需约 23 GB。分块不改变任何一条排名（每行独立）。
    """
    import torch
    dev_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    zL_t = torch.as_tensor(np.asarray(zL, np.float32), device=dev_name)
    zR_t = torch.as_tensor(np.asarray(zR, np.float32), device=dev_name)
    pairs = np.asarray(pairs, dtype=np.int64)
    cols = None
    out = {}
    for name, Zq, Zc, ti in (("region→phrase", zL_t, zR_t, 1),
                             ("phrase→region", zR_t, zL_t, 0)):
        arr = pairs[:, [0, 1]] if ti == 1 else pairs[:, [1, 0]]
        ranks = np.empty(len(arr), np.int64)
        for i in range(0, len(arr), chunk):
            blk = arr[i:i + chunk]
            S = Zq[torch.as_tensor(blk[:, 0], device=dev_name)] @ Zc.t()
            gt = S[torch.arange(len(blk), device=dev_name),
                   torch.as_tensor(blk[:, 1], device=dev_name)].unsqueeze(1)
            greater = (S > gt).sum(dim=1)
            eq = (S == gt)
            if cols is None or cols.shape[0] != S.shape[1]:
                cols = torch.arange(S.shape[1], device=dev_name).unsqueeze(0)
            before = (eq & (cols < torch.as_tensor(blk[:, 1], device=dev_name)
                            .unsqueeze(1))).sum(dim=1)
            ranks[i:i + chunk] = (greater + before + 1).cpu().numpy()
        out[name] = {f"R@{k}": float((ranks <= k).mean()) for k in ks}
        out[name]["MRR"] = float((1.0 / ranks).mean())
        out[name]["n_query"] = int(len(arr))
        out[name]["n_cand"] = int(Zc.shape[0])
        out[name]["rank_mean"] = float(ranks.mean())
        out[name]["rank_median"] = float(np.median(ranks))
    out["random_R@1"] = float(1.0 / max(zR.shape[0], 1))
    return out


def attach(pair, zL: np.ndarray, zR: np.ndarray, note: str = "") -> None:
    """把编码器嵌入写成两侧的**语义视图覆盖值**，供 `features.build_view` 使用。

    覆盖值必须已 L2 归一化（`features` 的其它视图都满足这一约定）。
    """
    def _n(M):
        M = np.asarray(M, dtype=np.float32)
        v = np.linalg.norm(M, axis=1, keepdims=True)
        v[v == 0] = 1.0
        return M / v
    pair.left.sem_override = _n(zL)                      # type: ignore[attr-defined]
    pair.right.sem_override = _n(zR)                     # type: ignore[attr-defined]
    pair.left.sem_override_note = note                   # type: ignore[attr-defined]
    pair.right.sem_override_note = note                  # type: ignore[attr-defined]
