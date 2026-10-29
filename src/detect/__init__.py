"""下游检测包。

子模块职责：
    constants   常量与下游图模式契约
    checkpoint  checkpoint 读写与部分加载
    fusion_model TFusionPPModel（backbone + 图增强跨模态融合）
    downstream  下游检测器装配与冻结表征抽取
    metrics     二分类检测指标
    thresholds  决策阈值选择与跨分布迁移
    validation  验证集切分与上下文缩放
    artifacts   评估产物与训练溯源落盘
    pretraining 预训练（对比损失）
    training    下游训练编排
"""
