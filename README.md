# CUSSM 实验代码与结果

本仓库是论文《基于范畴代数的跨模态知识统一建模与保结构语义匹配方法》第 5 章实验的  
**可执行对偶**：论文写的每一个协议、每一个指标、每一个数据获取步骤，在这里都有一段代码与之  
对应。仓库内所有数值结果均由代码实跑产生，不存在手工填入的数字。

方法名 **CUSSM**（Categorical algebra-based method for Unified modeling and Structure-preserving  
Semantic Matching of cross-modal knowledge）即论文中的本文方法；早期开发阶段曾用 `SCIM` 作为内部  
代号，本仓库已统一为 `CUSSM`（代码标识、结果键名、产物文件名一律一致）。

> 代码层面的技术说明见 [`code/README.md`](code/README.md)。

## 一、目录结构

```
cussm-release/
├── code/                 实验代码
│   ├── download_data.py  数据集下载层（镜像链 + SHA-1 校验 + 来源清单）
│   ├── run_experiment.py 统一入口（主表 / 消融 / 通道扫描 / 跨数据集）
│   ├── measure_ch5.py    第 5 章读数复算（效率、资源、覆盖面）
│   ├── measure_frontier.py  近五年方法对照读数
│   ├── cussm/            本文方法：统一建模 / 保结构匹配 / SPS / SIC
│   ├── baselines/        8 个基线（含自定基线、LinFuse、结构嵌入族、近五年方法）
│   ├── metrics/          指标与统计检验
│   └── README.md         代码层技术说明
├── data/
│   ├── manifest.json     49 个数据文件的来源、体积与 SHA-1 清单
│   └── raw/kg_small/     小规模自检集（约 3.4 MB，随仓库分发，便于离线冒烟）
└── results/              实跑产物（json / md / 排名矩阵）
```

## 二、快速开始

```bash
python -m pip install -r requirements.txt

# 1) 获取数据（原始数据不随仓库分发，见下一节）
python code/download_data.py --list          # 查看清单与体积
python code/download_data.py --group small   # 小规模自检集（约 3.4 MB，已随仓库附带）
python code/download_data.py --group dbp15k  # DBP15K 三语对（约 530 MB）
python code/download_data.py --group f30k    # Flickr30K Entities 标注（约 28 MB）
python code/download_data.py --group kgbench # FB15k-237 / WN18 / YAGO3-10（约 82 MB）
python code/download_data.py --verify        # 按 SHA-1 校验已下载文件

# 2) 离线冒烟（无需下载数据，用随仓库附带的 countries_S1 自检集，实测约 4 秒）
python code/run_experiment.py --datasets countries_s1 --max-test 50 --quick \
    --boot 100 --perm 200 --methods NNSim LinFuse CUSSM
#   ↑ 该命令的读数应与 results/selfcheck_countries_s1.json 逐位一致
#     （CUSSM / LinFuse / NNSim：Hits@1=1.0000、MRR=1.0000、纤维保持率=0.2400、SPS=0.7165）

# 3) 跑实验
python code/run_experiment.py --datasets dbp15k_fr_en --max-test 500 --quick --out results/smoke.json
python code/run_experiment.py --datasets dbp15k_fr_en      # 单数据集完整
python code/run_experiment.py                              # 全部数据集
```

## 三、数据从哪来

原始数据集**不随本仓库分发**（共约 643 MB，且均为第三方数据、各有其原始许可）。仓库只提供  
`code/download_data.py` 与 `data/manifest.json`：前者按 GitHub Contents API 的 **git blob SHA-1**  
做内容级校验并支持多镜像链与断点续传，后者记录 49 个文件的上游仓库、命中镜像、体积、SHA-1  
与下载耗时，可据此逐字节复核。

| 数据集                           | 上游仓库                                                   | 用途                 |
| ----------------------------- | ------------------------------------------------------ | ------------------ |
| DBP15K（fr_en / ja_en / zh_en） | `github.com/liuhaiyag/DBP15k_dataset`                  | Track B 跨语言实体对齐主表  |
| Flickr30K Entities            | `github.com/BryanPlummer/flickr30k_entities`           | Track A′ 区域—短语代理轨道 |
| FB15k-237 / WN18 / YAGO3-10   | `github.com/DeepGraphLearning/KnowledgeGraphEmbedding` | Track D 自对齐压力测试    |

各数据集的许可与引用遵循其原始出处；本仓库不主张对数据的任何权利。

## 四、结果文件说明

| 文件                              | 内容                                             |
| ------------------------------- | ---------------------------------------------- |
| `results/results.json`          | 主表与全部轨道的完整实跑结果（含运行环境、耗时、内存）                    |
| `results/force_ablation.json`   | 通道强制开启的消融台账                                    |
| `results/ch5_measure.json`      | 第 5 章效率与资源读数                                   |
| `results/ch5_frontier.json`     | 近五年方法对照读数                                      |
| `results/track_a_prime.json`    | Track A′ 代理轨道读数（论文该表的数据来源）                     |
| `results/recent/`               | 近五年方法（ICL / SelfKG / Dual-AMN / NeuSymEA）分方法读数 |
| `results/ranks/`                | 各方法在各数据集上的排名矩阵（`.npy`），供指标复核                   |
| `results/RESULTS.md`、`CH5_*.md` | 上述结果的 Markdown 台账                              |

结果中的运行环境字段已去除主机名与解释器绝对路径，仅保留 Python / NumPy / SciPy / Torch 版本与  
设备类型等与复现相关的信息。

## 五、未随仓库发布的内容

以下内容涉及运行环境凭据或中间态，已在本仓库中移除：

- 云主机运维脚本与手册（含主机地址、SSH 私钥文件名、登录用户名）；
- 本机上机自检产物（含硬件指纹、解释器绝对路径、BLAS 构建路径）；
- 运行日志与云端日志；
- 中间态备份文件（`*.bak`、`*prev*`、`*pre_offered*`）——仅保留最终口径的结果。

## 六、已知边界

见 [`code/README.md`](code/README.md) 第六节，其中关于 Track A / MMKG / ERNIE-ViL 的不可获取判定  
仍然有效；Track A′ 的处置以 `results/track_a_prime.json` 与论文对应表格为准。

## 七、许可

本仓库代码采用 **MIT License**，全文见 [`LICENSE`](LICENSE)。

```
MIT License

Copyright (c) 2026 syh8086
```

第三方数据集与第三方方法实现遵循其各自原始许可，见上文第三节。
