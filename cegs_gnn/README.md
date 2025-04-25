# CEGS GNN优化

本项目旨在增强CEGS（Configuration Example Generalizing Synthesizer）框架中的图神经网络（GNN）组件，特别是通过将边角色信息整合到GNN的消息传递机制中。

## 项目结构

- `models/`: GNN模型实现
  - `rgcn.py`: R-GCN (关系图卷积网络)实现
  - `edge_sage.py`: 边条件聚合的GraphSAGE实现
  - `baseline.py`: 基准GraphSAGE模型
- `data_utils/`: 数据处理工具
  - `data_loader.py`: 数据加载和预处理
  - `graph_utils.py`: 图数据操作工具
- `training/`: 训练相关代码
  - `trainer.py`: 模型训练循环
- `evaluation/`: 评估相关代码
  - `metrics.py`: 评估指标实现
  - `evaluator.py`: 模型评估循环

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

请参考`examples/`目录中的示例脚本和Jupyter笔记本。 