"""CUSSM —— 面向多模态知识统一建模的保结构匹配。

子模块：
    data        统一数据契约（第 5.4.4 节 I/O 契约的代码实现）
    features    结构与语义特征层（所有方法共用，保证"输入一致"）
    transport   Sinkhorn 最优传输与匈牙利指派
    model       本文方法 CUSSM（纤维化 lax 函子 + 路由融合）
    sps         SPS 三分量 Δ_comm / Δ_nat / Δ_glue 与纤维保持率
    sic         语义解释链 SIC 生成
"""
__all__ = ["data", "features", "transport", "model", "sps", "sic"]
