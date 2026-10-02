# -*- coding: utf-8 -*-
"""B 轨：近五年的结构侧实体对齐方法（2021–2022），在本文统一契约下**重实现**。

方案与纪律见 `.workbuddy/repro_plan_trackB_2026-09-30.md`。这些方法与
`baselines/models.py` 里的 TransE-NN／MTransE／JAPE 走**同一条路线**——
按其原文算法实现、用本文的统一训练/评测脚本重训，并在表 6 表注声明
「重训、不代表其正式发表性能」。

| 键 | 方法 | 出处 | 监督 | 输入特征 | 对齐 |
|---|---|---|---|---|---|
| `ICL`    | ICL/ICLEA | CIKM 2022, pp. 2465–2475 | 自监督（无种子） | LaBSE 名 | 伪对 Procrustes |
| `Dual-AMN` | Dual-AMN | WWW 2021, pp. 821–832 | 有种子 | 随机初始化嵌入 | 隐式（共享变换） |
| `SelfKG` | SelfKG | WWW 2022, pp. 860–870 | 自监督（无种子） | LaBSE 名 | 隐式（uni-space） |
| `NeuSymEA` | NeuSymEA | NeurIPS 2025 | 有种子（＋符号推理） | 随机初始化嵌入（神经侧＝Dual-AMN） | 隐式（共享变换） |
| `RNHGT`  | RPR-RHGT | IJCAI 2022, 3:1930–1937 | 有种子 | 名向量 + 路径 | —— |

**NeuSymEA 的落地形态**：官方栈为 `tensorflow 2.7 / keras 2.7`（仅支持 Python ≤ 3.9），
本机与云端（Py3.12、无 conda）均不可安装 ⇒ 同样按契约内重实现。其神经侧**就是
Dual-AMN**（官方 `config.py` 的 `ea_model` 默认值）故直接复用 `dual_amn.py`；符号侧
`probabilisticReasoning.py` 为纯 Python，逐行移植。**不属于**「同一预算」组：官方设计
是神经-符号迭代互注（`iter=5` × `epoch=20` + 每轮 10 轮符号推理），故**单列**，
并**按其原文默认配置**取 `--epochs 100`（＝ 5 轮互注 × 每轮 20 个神经轮），
不与前三条的 `--epochs 60` 作同预算并读。

**RNHGT 未实现**：其官方数据托管于坚果云且需预计算的路径文件
（`path_neigh_dict`／`rpath_sort_dict`），2026-09-30 实测两条分享链接
一条返回 HTTP 400、一条文件夹为空 ⇒ 按方案 §9-④ **以「官方数据不可得」为由
不纳入**，不以替代数据冒充（证据见 `FIDELITY.md`）。

逐方法的「忠实／简化构件」对照见本目录 `FIDELITY.md`。

**键名会进入 `results/ranks/*.npy` 的文件名** —— 禁用 Windows 非法字符
`< > : " / \\ | ? *`（曾因此落盘失败）。
"""
from .dual_amn import DualAMN
from .icl import ICL
from .neusymea import NeuSymEA
from .selfkg import SelfKG

# 键 = 结果表用的短 id（同时也是 ranks 文件名、`--methods` 过滤键）
RECENT_BASELINES = {
    "ICL": ICL,
    "Dual-AMN": DualAMN,
    "SelfKG": SelfKG,
    "NeuSymEA": NeuSymEA,
}

__all__ = ["ICL", "DualAMN", "SelfKG", "NeuSymEA", "RECENT_BASELINES"]
