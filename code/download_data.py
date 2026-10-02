#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CUSSM 第 5 章 —— 开源数据集下载层。

设计要点（对应第 5.1 节"获取与核验"）：
  1. **多镜像链**：先按实测速度排序的镜像逐个尝试，单个镜像失败即自动切换；
  2. **断点续传**：curl -C - ，镜像切换后从已落盘字节继续（需两镜像字节一致）；
  3. **完整性校验**：以 GitHub Contents API 返回的 git blob SHA-1 为基准，
     本地用 sha1(b"blob %d\\0" % size + content) 复算比对 —— 这是内容级校验，
     不是只看体积；GitHub API 还能顺带给出体积，两者都必须吻合；
  4. **来源清单**：把实际命中的镜像、体积、SHA-1、下载耗时写入 data/manifest.json，
     供第 5 章"数据获取 provenance"与附录引用。

用法：
    python code/download_data.py --list                 # 列出全部数据集
    python code/download_data.py --group dbp15k         # 下载 DBP15K 三语对
    python code/download_data.py --group f30k           # 下载 Flickr30K Entities
    python code/download_data.py --group small          # 下载小规模 KG（countries/wn18rr）
    python code/download_data.py --group kgbench        # 下载中等规模 KG（FB15k-237/wn18/YAGO3-10）
    python code/download_data.py                        # 下载全部（6 个数据集方向）
    python code/download_data.py --verify               # 只校验已有文件
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
MANIFEST = os.path.join(ROOT, "data", "manifest.json")

# ---------------------------------------------------------------- 镜像链
# 2026-09-27 实测速度（对 1.45MB 文件）：gh-proxy 502KB/s、fastly.jsdelivr 380KB/s、
# gh.xxooo.cf 349KB/s、gcore.jsdelivr 102KB/s、raw.githubusercontent 58KB/s。
# 已确认失效者不列入：github.com 直连(clone 被 reset)、raw.githack、hub.gitmirror、
# ghproxy.cc、ghproxy.net、statically、gcore(不稳)。
MIRRORS = [
    ("gh-proxy.com", "https://gh-proxy.com/https://raw.githubusercontent.com/{repo}/{branch}/{path}"),
    ("fastly.jsdelivr.net", "https://fastly.jsdelivr.net/gh/{repo}@{branch}/{path}"),
    ("gh.xxooo.cf", "https://gh.xxooo.cf/https://raw.githubusercontent.com/{repo}/{branch}/{path}"),
    ("gcore.jsdelivr.net", "https://gcore.jsdelivr.net/gh/{repo}@{branch}/{path}"),
    ("raw.githubusercontent.com", "https://raw.githubusercontent.com/{repo}/{branch}/{path}"),
]

# ---------------------------------------------------------------- 数据集登记
# **严禁在此手写 sha** —— 手写值一旦与实际不符，会让校验必然失败并把已下好的文件删掉
# （2026-09-27 实测踩过此坑：fr_en/ent_ILLs 因臆测 sha 被误删）。
# `size` 一栏是 2026-09-27 由 GitHub Contents API 实取的权威体积，作为**离线也可用**的
# 校验基准（API 会限流，见 --deep-verify 说明）；sha 一律留给 API 在运行时补。
DATASETS = {
    # ---- DBP15K：跨语言知识图谱实体对齐基准（Sun et al., ISWC 2017）----
    "dbp15k_fr_en": {
        "group": "dbp15k", "repo": "liuhaiyag/DBP15k_dataset", "branch": "main",
        "note": "DBP15K French-English 跨语言实体对齐基准",
        "files": {
            "ent_ILLs":       (1452232, None),
            "en_rel_triples": (36611053, None), "fr_rel_triples": (27368768, None),
            "en_att_triples": (73207089, None), "fr_att_triples": (69874563, None),
        },
    },
    "dbp15k_zh_en": {
        "group": "dbp15k", "repo": "liuhaiyag/DBP15k_dataset", "branch": "main",
        "note": "DBP15K Chinese-English 跨语言实体对齐基准",
        "files": {
            "ent_ILLs": (1429971, None),
            "en_rel_triples": (30861428, None), "zh_rel_triples": (21202268, None),
            "en_att_triples": (71790674, None), "zh_att_triples": (50971318, None),
        },
    },
    "dbp15k_ja_en": {
        "group": "dbp15k", "repo": "liuhaiyag/DBP15k_dataset", "branch": "main",
        "note": "DBP15K Japanese-English 跨语言实体对齐基准",
        "files": {
            "ent_ILLs": (1594990, None),
            "en_rel_triples": (30321256, None), "ja_rel_triples": (25708745, None),
            "en_att_triples": (62547286, None), "ja_att_triples": (51141019, None),
        },
    },
    # ---- Flickr30K Entities：多模态实体提及与共指链（Plummer et al., ICCV 2015）----
    "flickr30k_entities": {
        "group": "f30k", "repo": "BryanPlummer/flickr30k_entities", "branch": "master",
        "note": "Flickr30K Entities 标注包（每图的实体提及、共指链、边界框）",
        "files": {"annotations.zip": (29284070, None),
                  "train.txt": (320344, None), "val.txt": (10758, None),
                  "test.txt": (10733, None)},
    },
    # ---- 小规模 KG 基准（DeepGraphLearning, MIT）：用于稠密图上的快速复现----
    "kg_small": {
        "group": "small", "repo": "DeepGraphLearning/KnowledgeGraphEmbedding", "branch": "master",
        "note": "小规模 KG 基准，用于紧凑实验与单元自检",
        "files": {
            "data/countries_S1/train.txt":  (0, None),
            "data/countries_S1/valid.txt":  (0, None),
            "data/countries_S1/test.txt":   (0, None),
            "data/countries_S2/train.txt":  (0, None),
            "data/countries_S2/valid.txt":  (0, None),
            "data/countries_S2/test.txt":   (0, None),
            "data/countries_S3/train.txt":  (0, None),
            "data/countries_S3/valid.txt":  (0, None),
            "data/countries_S3/test.txt":   (0, None),
            "data/wn18rr/train.txt":        (0, None),
            "data/wn18rr/valid.txt":        (0, None),
            "data/wn18rr/test.txt":         (0, None),
        },
    },
    # ---- 中等规模标准 KG 基准（同上仓库，MIT）：Track D 结构与规模的对照 ----
    # 这三个是 KG 嵌入/链接预测的公认基准（Bordes et al. 2013; Toutanova & Chen 2015;
    # Mahdisoltani et al. 2015），在此登记为**补充公开数据集**，为第 5 章提供
    # 6 个以上真实数据源与不同规模档位。
    "fb15k237": {
        "group": "kgbench", "repo": "DeepGraphLearning/KnowledgeGraphEmbedding",
        "branch": "master", "note": "FB15k-237（Freebase 子集，链接预测标准基准）",
        "files": {
            "data/FB15k-237/train.txt":      (0, None),
            "data/FB15k-237/valid.txt":      (0, None),
            "data/FB15k-237/test.txt":       (0, None),
            "data/FB15k-237/entities.dict":  (0, None),
            "data/FB15k-237/relations.dict": (0, None),
            "data/FB15k-237/README.txt":     (0, None),
        },
    },
    "wn18": {
        "group": "kgbench", "repo": "DeepGraphLearning/KnowledgeGraphEmbedding",
        "branch": "master", "note": "WN18（WordNet 子集，链接预测标准基准）",
        "files": {
            "data/wn18/train.txt":            (0, None),
            "data/wn18/valid.txt":            (0, None),
            "data/wn18/test.txt":             (0, None),
            "data/wn18/entities.dict":        (0, None),
            "data/wn18/relations.dict":       (0, None),
            "data/wn18/README":               (0, None),
            "data/wn18/Wordnet3.0-LICENSE":   (0, None),
        },
    },
    "yago3_10": {
        "group": "kgbench", "repo": "DeepGraphLearning/KnowledgeGraphEmbedding",
        "branch": "master", "note": "YAGO3-10（YAGO 子集，链接预测标准基准）",
        "files": {
            "data/YAGO3-10/train.txt":      (0, None),
            "data/YAGO3-10/valid.txt":      (0, None),
            "data/YAGO3-10/test.txt":       (0, None),
            "data/YAGO3-10/entities.dict":  (0, None),
            "data/YAGO3-10/relations.dict": (0, None),
        },
    },
}

API = "https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"


def _git_blob_sha1(data: bytes) -> str:
    """复算 git blob SHA-1：sha1(b'blob <len>\\x00' + content)。"""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def api_meta(repo: str, branch: str, path: str, retry: int = 3) -> dict | None:
    """取 GitHub Contents API 元信息（size 与 git blob sha）。"""
    url = API.format(repo=repo, path=path, branch=branch)
    for i in range(retry):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cussm-downloader"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            if i == retry - 1:
                print(f"    [api] {path}: {exc}")
                return None
            time.sleep(2 + 2 * i)
    return None


def curl(url: str, dest: str, resume: bool, timeout: int = 900) -> tuple[bool, int, float]:
    """单次 curl 拉取。返回 (成功, 已落盘字节, 用时)。"""
    cmd = ["curl", "-sL", "--fail", "--retry", "2", "--retry-delay", "2",
           "-m", str(timeout), "-o", dest, "-w", "%{size_download} %{speed_download}"]
    if resume:
        cmd += ["-C", "-"]
    cmd.append(url)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    return p.returncode == 0, size, dt


def _url_path(ds_key: str, fname: str) -> str:
    """文件在 GitHub 仓库中的路径。"""
    if ds_key.startswith("dbp15k"):
        return f"{ds_key.split('_', 1)[1]}/{fname}"      # fr_en/ent_ILLs
    return fname                                          # annotations.zip / data/...


def fetch(ds_key: str, fname: str, spec: tuple, force: bool = False) -> dict:
    ds = DATASETS[ds_key]
    rel = fname
    url_path = _url_path(ds_key, fname)

    dest = os.path.join(RAW, ds_key, *rel.split("/"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # 1) 从 API 取权威 size / sha（也用于补全未登记项）
    meta = api_meta(ds["repo"], ds["branch"], url_path)
    exp_size, exp_sha = (meta.get("size"), meta.get("sha")) if meta else (spec[0], spec[1])
    if spec[1]:
        exp_sha = spec[1]

    # 2) 已存在且校验通过 → 跳过
    if os.path.exists(dest) and not force and exp_sha:
        with open(dest, "rb") as f:
            if _git_blob_sha1(f.read()) == exp_sha:
                print(f"  [skip] {ds_key}/{rel}  已通过 SHA-1 校验 ({exp_size} B)")
                return {"bytes": exp_size, "sha1": exp_sha, "mirror": "(cached)", "ok": True}

    # 3) 多镜像逐个尝试
    for mname, tpl in MIRRORS:
        url = tpl.format(repo=ds["repo"], branch=ds["branch"], path=url_path)
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        print(f"  [{mname}] {ds_key}/{rel}  (已落盘 {have}/{exp_size} B)")
        ok, size, dt = curl(url, dest, resume=have > 0)
        if not ok and size == 0:
            continue
        # 4) 校验：体积
        if exp_size and size != exp_size:
            print(f"    ✗ 体积不符 {size} != {exp_size}，换镜像续传")
            continue
        # 5) 校验：git blob SHA-1
        with open(dest, "rb") as f:
            act = _git_blob_sha1(f.read())
        if exp_sha and act != exp_sha:
            print(f"    ✗ SHA-1 不符 {act[:12]} != {exp_sha[:12]}，重下")
            os.remove(dest)
            continue
        spd = size / dt / 1024 if dt else 0
        print(f"    ✓ {size} B  {spd:.0f} KB/s  用时 {dt:.1f}s  sha1={act[:12]}")
        return {"bytes": size, "sha1": act, "mirror": mname, "ok": True,
                "seconds": round(dt, 2), "kBps": round(spd, 1)}

    print(f"    ✗✗ 全部镜像失败：{ds_key}/{rel}")
    return {"bytes": 0, "sha1": None, "mirror": None, "ok": False}


def load_manifest() -> dict:
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            return json.load(f)
    return {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "items": {}}


def save_manifest(m: dict) -> None:
    m["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--group", default=None, help="dbp15k | f30k | small | kgbench")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--deep-verify", action="store_true",
                    help="向 GitHub API 复核每个文件的 git blob SHA-1（会消耗 API 配额）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="仅下某文件（调试用）")
    a = ap.parse_args()

    if a.deep_verify:
        man = load_manifest()
        n_ok, n_bad, n_skip = 0, [], 0
        for key, it in man.get("items", {}).items():
            k, fname = key.split("/", 1)
            ds = DATASETS.get(k)
            if not ds:
                continue
            meta = api_meta(ds["repo"], ds["branch"], _url_path(k, fname))
            if not meta:
                n_skip += 1
                print(f"  ?  {key}: API 不可用（限流 / 登记分支或路径已变更），跳过")
                continue
            dest = os.path.join(RAW, k, *fname.split("/"))
            if not os.path.exists(dest):
                print(f"  ✗  {key}: 本地文件缺失")
                n_bad.append(key)
                continue
            # **比对对象必须是本地内容**：旧实现拿 API 的 sha 与"清单里记的 sha"比，
            # 那只证明"清单没被改"，证明不了"磁盘上的文件没被改"。深度校验若不读盘，
            # 就与 --verify 同义。
            with open(dest, "rb") as f:
                act = _git_blob_sha1(f.read())
            size = os.path.getsize(dest)
            same = (meta.get("sha") == act)
            size_ok = (meta.get("size") == size)
            it["api_sha"] = meta["sha"]
            it["api_size"] = meta["size"]
            it["bytes"] = size
            it["sha1"] = act
            it["ok"] = bool(same and size_ok)
            it["deep_checked"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            print(f"  {'✓' if it['ok'] else '✗'}  {key}  api={str(meta.get('sha'))[:12]} "
                  f"local={act[:12]} size={meta.get('size')}/{size}"
                  f"{'' if size_ok else '  ← 体积不符!'}")
            if it["ok"]:
                n_ok += 1
            else:
                n_bad.append(key)
        save_manifest(man)
        print(f"=== 深度校验：{n_ok} 项与 GitHub 权威值一致，{len(n_bad)} 项不符，"
              f"{n_skip} 项跳过 ===")
        if n_bad:
            print("  不符：", n_bad)
        return 1 if n_bad else 0

    if a.list:
        for k, v in DATASETS.items():
            tot = sum(s for s, _ in v["files"].values())
            print(f"  {k:<22} [{v['group']:<6}] {len(v['files'])} 文件  "
                  f"{tot/1048576:.1f} MB  {v['note']}")
        return 0

    targets = [k for k, v in DATASETS.items()
               if (a.group is None or v["group"] == a.group)
               and (a.dataset is None or k == a.dataset)]
    if not targets:
        print("没有匹配的数据集；用 --list 查看")
        return 2

    man = load_manifest()
    print(f"=== 开始下载 {len(targets)} 个数据集 → {RAW} ===")
    nfail = 0
    for k in targets:
        print(f"- {k}: {DATASETS[k]['note']}")
        for fname, spec in DATASETS[k]["files"].items():
            if a.only and a.only not in fname:
                continue
            if a.verify:
                dest = os.path.join(RAW, k, *fname.split("/"))
                if not os.path.exists(dest):
                    print(f"  [miss] {k}/{fname}")
                    nfail += 1
                    continue
                continue
            res = fetch(k, fname, spec, force=a.force)
            man["items"][f"{k}/{fname}"] = {
                "dataset": k, "repo": DATASETS[k]["repo"], **res,
                "checked": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            save_manifest(man)
            if not res["ok"]:
                nfail += 1

    if a.verify:
        print("=== 校验 ===")
        n_missing = n_unver = n_changed = 0
        for key, it in man.get("items", {}).items():
            k, fname = key.split("/", 1)
            dest = os.path.join(RAW, k, *fname.split("/"))
            if not os.path.exists(dest):
                print(f"  ✗ 缺失 {key}")
                n_missing += 1
                continue
            # **比对基准必须是"上游权威 SHA-1"（`api_sha`），不是清单里的 `sha1`。**
            # 2026-09-28 实测踩过：`--deep-verify` 会把**本地内容哈希**回填进 `sha1`
            # （见该分支的 `it["sha1"] = act`）。若 `--verify` 仍拿 `sha1` 当基准，
            # 就成了"本地 vs 本地"的恒等比较 —— 7 个未取到权威值的条目会**假通过**，
            # 打出 "49/49 项通过"。故此处只认 `api_sha`。
            auth = it.get("api_sha")
            if not auth:
                print(f"  ? 未核验 {key}：清单无上游权威 SHA-1，请先跑 --deep-verify 补录")
                n_unver += 1
                continue
            with open(dest, "rb") as f:
                act = _git_blob_sha1(f.read())
            if act != auth:
                print(f"  ✗ 变更 {key}  {act[:12]} != {auth[:12]}")
                n_changed += 1
        tot = len(man.get("items", {}))
        bad = n_missing + n_unver + n_changed
        print(f"  {tot - bad}/{tot} 项通过（缺失 {n_missing}，无权威 SHA-1 未核验 {n_unver}，"
              f"内容变更 {n_changed}）")
        return 1 if bad else 0

    print(f"=== 完成，失败 {nfail} 项 ===")
    print(f"来源清单：{MANIFEST}")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
