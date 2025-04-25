import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

# 添加父目录到sys.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型和工具
from models.baseline import BaselineSAGE
from models.rgcn import RGCN
from models.edge_sage import EdgeCondSAGE
from data_utils.data_loader import load_synthetic_data, prepare_data_for_models
from training.trainer import ModelTrainer
from evaluation.evaluator import ModelEvaluator
from evaluation.metrics import accuracy


def main():
    print("CEGS GNN训练测试")
    
    # 设置参数
    node_dim = 16
    edge_dim = 8
    hidden_dim = 32
    out_dim = 16
    num_epochs = 3
    batch_size = 16
    learning_rate = 0.005
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    print("\n准备数据...")
    base_dataset, rgcn_dataset, edge_sage_dataset, num_relations = prepare_data_for_models(
        root="./data",
        load_synthetic=True,
        embedding_dim=edge_dim
    )
    print(f"数据集大小: {len(base_dataset)}")
    print(f"关系类型数量: {num_relations}")
    
    # 数据分割
    torch.manual_seed(42)
    n = len(base_dataset)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    indices = torch.randperm(n)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    baseline_train_loader = DataLoader([base_dataset[i] for i in train_indices], batch_size=batch_size, shuffle=True)
    baseline_val_loader = DataLoader([base_dataset[i] for i in val_indices], batch_size=batch_size, shuffle=False)
    baseline_test_loader = DataLoader([base_dataset[i] for i in test_indices], batch_size=batch_size, shuffle=False)
    
    rgcn_train_loader = DataLoader([rgcn_dataset[i] for i in train_indices], batch_size=batch_size, shuffle=True)
    rgcn_val_loader = DataLoader([rgcn_dataset[i] for i in val_indices], batch_size=batch_size, shuffle=False)
    
    edge_sage_train_loader = DataLoader([edge_sage_dataset[i] for i in train_indices], batch_size=batch_size, shuffle=True)
    edge_sage_val_loader = DataLoader([edge_sage_dataset[i] for i in val_indices], batch_size=batch_size, shuffle=False)
    
    # 创建模型
    print("\n创建模型...")
    # 1. 基准GraphSAGE模型
    baseline_model = BaselineSAGE(
        in_channels=node_dim,
        hidden_channels=hidden_dim,
        out_channels=out_dim,
        num_layers=2,
        dropout=0.1
    )
    
    # 2. R-GCN模型
    rgcn_model = RGCN(
        in_channels=node_dim,
        hidden_channels=hidden_dim,
        out_channels=out_dim,
        num_relations=num_relations,
        num_layers=2,
        dropout=0.1,
        num_bases=4
    )
    
    # 3. EdgeCondSAGE模型
    edge_sage_model = EdgeCondSAGE(
        in_channels=node_dim,
        hidden_channels=hidden_dim,
        out_channels=out_dim,
        edge_dim=edge_dim,
        num_layers=2,
        dropout=0.1
    )
    
    # 创建训练器
    print("\n创建训练器...")
    baseline_trainer = ModelTrainer(
        model=baseline_model,
        device=device,
        task_type='node',
        lr=learning_rate
    )
    
    rgcn_trainer = ModelTrainer(
        model=rgcn_model,
        device=device,
        task_type='node',
        lr=learning_rate
    )
    
    edge_sage_trainer = ModelTrainer(
        model=edge_sage_model,
        device=device,
        task_type='node',
        lr=learning_rate
    )
    
    # 训练模型（简短训练用于演示）
    print("\n训练基准模型...")
    baseline_trainer.train(
        train_loader=baseline_train_loader,
        val_loader=baseline_val_loader,
        num_epochs=num_epochs,
        metric_fn=accuracy
    )
    
    print("\n训练R-GCN模型...")
    rgcn_trainer.train(
        train_loader=rgcn_train_loader,
        val_loader=rgcn_val_loader,
        num_epochs=num_epochs,
        metric_fn=accuracy
    )
    
    print("\n训练EdgeCondSAGE模型...")
    edge_sage_trainer.train(
        train_loader=edge_sage_train_loader,
        val_loader=edge_sage_val_loader,
        num_epochs=num_epochs,
        metric_fn=accuracy
    )
    
    # 评估模型
    print("\n评估模型...")
    evaluator = ModelEvaluator(task_type='node')
    evaluator.add_model('Baseline', baseline_trainer)
    evaluator.add_model('RGCN', rgcn_trainer)
    evaluator.add_model('EdgeSAGE', edge_sage_trainer)
    
    # 模拟Querier任务
    print("\n评估Querier任务（示例推荐）...")
    querier_results = evaluator.evaluate_example_recommendation(baseline_test_loader)
    
    # 打印参数数量比较
    print("\n模型参数量比较:")
    for name, result in querier_results.items():
        print(f"{name}: {result['parameters']:,} 参数")
    
    print("\n测试完成!")


if __name__ == "__main__":
    main() 