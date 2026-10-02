"""基线方法与其中的消融变体。

`baselines.models` 提供与本文方法**同输入、同输出**的 8 个对照方法，覆盖
"表层字面 / 属性语义 / 结构传播（不迭代）/ 结构传播（迭代自训练）/ 两视图线性融合 /
结构嵌入（独立训练+Procrustes）/ 结构嵌入（联合训练）/ 结构+属性联合嵌入" 五个梯队。

消融并非另写实现，而是通过 `cussm.model.CUSSM` 的构造开关产生，
确保主表与消融表跑的是同一条代码路径。
"""
from .models import (BASELINES, AttrSim, JAPE, LinFuse, MTransE, NNSim,
                     StructPropK, StructSim, TransENN)

__all__ = ["NNSim", "AttrSim", "StructSim", "StructPropK", "LinFuse",
           "TransENN", "MTransE", "JAPE", "BASELINES"]
