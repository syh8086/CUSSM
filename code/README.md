# 第 5 章实验代码（CUSSM）

本目录是论文第 5 章「实验方案」的**可执行对偶**：第 5 章写的每一个协议、每一个
指标、每一个获取步骤，在这里都有一段代码与之对应。所有与本目录有关的结果都由
`run_experiment.py` 实跑产生，不存在手工填入的数值。

## 环境

```bash
# 本机已验证的运行时
python   # Python 3.13.14
# 依赖（定版见 code/requirements.txt）：
#   numpy  2.5.3    —— 模型、指标、SPS、SIC、最优传输、统计检验全部基于它
#   scipy  1.18.1   —— 稀疏矩阵（scipy.sparse）与 linear_sum_assignment（匈牙利指派）
#   torch  2.6.0    —— 只被"结构嵌入族"（TransE-NN / MTransE / JAPE）与公共训练器
#                      cussm/torch_backend.py 使用；本机装的是 +cpu 版
python -m pip install -r code/requirements.txt
```

**硬件约束**：本机实验在 CPU-only 环境完成（无 CUDA 可用 GPU），且结构嵌入族的训练带
`--budget-mode wall --max-minutes 3` 的墙上时间上限，因此那三个基线的 `epochs_done` 小于规划的 60 轮。
`--budget-mode` 是显式开关（默认 `epochs`，即跑满 `--epochs`）；两档互斥校验：`epochs` 档传
`--max-minutes` 会报错，`wall` 档缺 `--max-minutes` 也报错。任一档下 `epochs_done` 明显小于规划轮数时都会打印醒目警告。

**关于云上运行**：开发期使用的云主机运维脚本与手册（含主机地址、SSH 私钥文件名、
登录用户名）属于环境凭据，未随本仓库发布。在任意装有 CUDA 的机器上按本文第二节的命令
即可复现，无需额外的上云流程。

## 一、下载数据

```bash
python code/download_data.py --list          # 查看数据集清单与体积
python code/download_data.py --group dbp15k  # DBP15K 三语对（约 530 MB）
python code/download_data.py --group f30k    # Flickr30K Entities 标注（约 28 MB）
python code/download_data.py --group small   # 小规模 KG（自检用）
python code/download_data.py --verify        # 校验已下文件
python code/download_data.py --deep-verify   # 向 GitHub API 复核 git blob SHA-1
```

下载层特点：

| 机制 | 说明 |
|---|---|
| 多镜像链 | `gh-proxy.com` → `fastly.jsdelivr.net` → `gh.xxooo.cf` → `gcore.jsdelivr.net` → `raw.githubusercontent.com`，逐个尝试，失败自动切换 |
| 断点续传 | `curl -C -`，镜像切换后从已落盘字节继续 |
| 完整性校验 | 以 GitHub Contents API 的 **git blob SHA-1** 为基准，本地按 `sha1(b"blob <len>\0" + content)` 复算比对；体积同时比对 |
| 来源清单 | 实际命中的镜像、体积、SHA-1、耗时写入 `data/manifest.json` |

> **实测说明（2026-09-27）**：`huggingface.co`、`zenodo.org`、`data.dws.informatik.uni-mannheim.de`
> （MMKG 官方站）、`github.com` 直连（clone）均不可达；`raw.githubusercontent.com`
> 可用但极慢；`gh-proxy.com` 实测 3–6 MB/s，是本次唯一稳定的大文件通路。

## 二、跑实验

```bash
# 冒烟（快）
python code/run_experiment.py --datasets dbp15k_fr_en --max-test 500 --quick --out results/smoke.json

# 单数据集完整（跑满 60 轮，等轮数口径）
python code/run_experiment.py --datasets dbp15k_fr_en

# 全部数据集（默认）
python code/run_experiment.py
```

产物：`results/results.json`、`results/ranks/*.npy`、`results/RESULTS.md`。

## 三、代码结构

```
code/
├── download_data.py          数据集下载层（镜像链 + SHA-1 校验 + 来源清单）
└── run_experiment.py         统一入口
```

## 四、输入输出一致性怎么被保证

不是靠约定，而是靠接口：

1. **输入**：所有方法只吃 `data.LOADERS[...]()` 返回的同一个 `Pair`；特征只由
   `features.build_view` 产出。任何方法若想自带特征，都得绕开接口，而接口是唯一的
   得分来源。
2. **划分**：`Pair` 内部已把 `seeds / val / test` 切好并保证测试实体不出现在
   训练与验证中；方法拿不到别的划分。
3. **输出**：所有方法只实现 `score_block(pair, left_idx)`（左侧子集 → 对右侧全部
   候选的得分）。排名与指标由 `metrics.core.evaluate` 统一计算，**连分块大小都相同**。
4. **调参**：所有方法的超参只在 `pair.val` 上选，测试集不参与任何选择（代码中
   `_tune` 与 `BootEA.fit` 都显式只用 `pair.val`）。

## 五、方法—公式对应

| 第 4 章构造 | 代码位置 |
|---|---|
| 定义 1 模态知识范畴 | `data.KG` |
| 定义 3 打字投影 π | `features.fibers_from`（两支合起来 k-means，共享类型空间） |
| 定义 5(2) 纤维保持 π_B∘F=π_A | `model.CUSSM._penalty`（**传输/打分前**把跨纤维配对压低） |
| 两条路由 H:=F_TK、K:=F_IK∘F_TI | `CUSSM._route_H` / `CUSSM._route_K` |
| 余极限粘合 pushout | `CUSSM._route_C`（合成映射 Ω_c 的双读数之和） |
| 4.3.1 SPS 聚合式 | `sps.audit` + `sps.aggregate`（β=1，α 由量纲归一标定） |
| 4.5 语义解释链 SIC | `sic.build_sic`（五步链 + C1–C4 校验） |

## 六、已知边界（写进第 5 章 5.7 节）

1. **Track A（图像-文本检索）不可执行**：本机无 GPU、无 HuggingFace 通路，CLIP/BLIP/ALBEF
   权重无法获取（换到 GPU 机器也解决不了 HuggingFace 通路未确证这一点，见
   （未随本仓库发布））；COCO / Flickr30K 图像达数 GB 且需授权。原拟以 Flickr30K Entities 的
   「区域↔短语」结构对齐作为 **Track A′ 代理**，但**实跑后判定其在本原型下同样不可评估**：
   左侧区域图不含关系三元组（结构画像恒空、结构覆盖率 0）、两侧特征分属不共享的两个
   空间、两侧索引相同且真值为恒等映射。故 A′ 与 A 一并归入「本轮不可评估」，**不进入
   任何数值表、也不给出其路由数值**。其加载器仍保留，并已修好两处缺陷（见 `cussm/data.py`）。
   可执行的主表轨道是 **Track B**（DBP15K 跨语言实体对齐）与 **Track D**（自对齐压力测试）。
2. **MMKG（FB15k/DB15k/Yago15k）不可获取**：官方站整站 403。
3. **ERNIE-ViL 代码路径未确证**：`PaddlePaddle/ERNIE` 已转向 4.5，`research/ernie-vil`
   返回 404。
4. **sheaf_kg 归属未确证**：`tgebhart/sheaf_kg` 无描述、无许可、星数极低。
5. **指标口径**：所有排名都在**完整候选集**上计算，不做候选截断，因此与使用
   ITM 重排 / top-k 截断的公开数值（ALBEF/BLIP 等）**不可直接并列**。
