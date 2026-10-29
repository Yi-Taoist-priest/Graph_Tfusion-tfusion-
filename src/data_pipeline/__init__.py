"""数据管线包。

子模块职责：
    constants  共享常量与最小样本约束
    reading    单文件 / 预切分 JSON 读取
    split      预切分类别先验告警
    features   模态特征构造（包序列、流级统计、主机因果统计）
    batching   对比学习预训练 batch 构造
    context    因果上下文特征与邻居检索
    loader     张量束组装入口 load_graph_tfusion_data
"""
