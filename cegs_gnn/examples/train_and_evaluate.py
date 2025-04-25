import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib
import matplotlib.pyplot as plt

# # 配置matplotlib显示中文 - 已移除
# try:
#     plt.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'Microsoft YaHei'] # 优先使用苹方，备选黑体、雅黑
#     plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# except Exception as e:
#     print(f"无法设置中文字体，可能需要手动安装字体: {e}")
#     # 如果找不到指定字体，可以尝试系统默认字体，但不一定支持中文
#     pass 

# 导入自定义模块
from ..models.baseline import BaselineSAGE
from ..models.rgcn import RGCN
from ..models.edge_sage import EdgeCondSAGE
from ..data_utils.data_loader import prepare_data_for_models
from ..training.trainer import ModelTrainer
from ..evaluation.evaluator import ModelEvaluator
from ..evaluation.metrics import accuracy, f1


def train_and_evaluate_models(root_dir="./data", output_dir="./results", 
                             edge_dim=8, hidden_dim=64, 
                             num_epochs=50, batch_size=32, learning_rate=0.001,
                             device=None):
    """
    训练和评估模型
    
    Args:
        root_dir (str): 数据根目录
        output_dir (str): 输出目录
        edge_dim (int): 边特征维度
        hidden_dim (int): 隐藏层维度
        num_epochs (int): 训练轮数
        batch_size (int): 批次大小
        learning_rate (float): 学习率
        device (torch.device, optional): 计算设备
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    print("准备数据...")
    base_dataset, rgcn_dataset, edge_sage_dataset, num_relations, num_classes, feature_dim = prepare_data_for_models(
        root=root_dir,
        load_synthetic=True,
        use_cegs_like_synth=True, # 确保使用新数据
        embedding_dim=edge_dim
        # num_classes 和 feature_dim 由函数内部确定并返回
    )
    
    # --- 查看生成的数据 --- 
    print("\n--- 查看第一个图的数据 ---")
    first_graph_base = base_dataset[0]
    first_graph_rgcn = rgcn_dataset[0]
    first_graph_edge_sage = edge_sage_dataset[0]
    
    print(f"Base Dataset - Graph 0:")
    print(f"  Number of nodes: {first_graph_base.num_nodes}")
    print(f"  Number of edges: {first_graph_base.num_edges}")
    print(f"  Node features shape (x): {first_graph_base.x.shape}")
    # print(f"  Node features (first 2):\n{first_graph_base.x[:2]}") # 可选：打印前几个特征
    print(f"  Edge index shape: {first_graph_base.edge_index.shape}")
    # print(f"  Edge index (first 5 pairs):\n{first_graph_base.edge_index[:, :5]}") # 可选：打印前几条边
    if hasattr(first_graph_base, 'y'):
        print(f"  Node labels shape (y): {first_graph_base.y.shape}")
        print(f"  Node labels (unique): {torch.unique(first_graph_base.y)}")
        # print(f"  Node labels (first 10): {first_graph_base.y[:10]}") # 可选：打印前几个标签
    else:
        print("  Node labels (y): Not found")
    if hasattr(first_graph_base, 'edge_roles'):
        print(f"  Edge roles (sample): {first_graph_base.edge_roles[:5]}...") # 打印前几个原始角色
    else:
        print("  Edge roles: Not found in base data object (this is unexpected)")

    print(f"\nRGCN Dataset - Graph 0:")
    if hasattr(first_graph_rgcn, 'edge_type'):
        print(f"  Edge type shape: {first_graph_rgcn.edge_type.shape}")
        print(f"  Edge type (unique): {torch.unique(first_graph_rgcn.edge_type)}")
        # print(f"  Edge type (first 5): {first_graph_rgcn.edge_type[:5]}") # 可选：打印前几个类型
    else:
        print("  Edge type: Not found")
        
    print(f"\nEdgeSAGE Dataset - Graph 0:")
    if hasattr(first_graph_edge_sage, 'edge_attr'):
        print(f"  Edge attributes shape: {first_graph_edge_sage.edge_attr.shape}")
        # print(f"  Edge attributes (first 2):\n{first_graph_edge_sage.edge_attr[:2]}") # 可选：打印前几个属性
    else:
        print("  Edge attributes: Not found")
    print("--- 数据查看结束 ---\n")
    # --- 查看结束 ---

    # 按照8:1:1分割数据集
    torch.manual_seed(42)
    
    # 计算分割点
    n = len(base_dataset)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    test_size = n - train_size - val_size
    
    # 随机打乱数据集
    indices = torch.randperm(n)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # 创建数据加载器
    baseline_train_loader = DataLoader([base_dataset[i] for i in train_indices], batch_size=batch_size, shuffle=True)
    baseline_val_loader = DataLoader([base_dataset[i] for i in val_indices], batch_size=batch_size, shuffle=False)
    baseline_test_loader = DataLoader([base_dataset[i] for i in test_indices], batch_size=batch_size, shuffle=False)
    
    rgcn_train_loader = DataLoader([rgcn_dataset[i] for i in train_indices], batch_size=batch_size, shuffle=True)
    rgcn_val_loader = DataLoader([rgcn_dataset[i] for i in val_indices], batch_size=batch_size, shuffle=False)
    rgcn_test_loader = DataLoader([rgcn_dataset[i] for i in test_indices], batch_size=batch_size, shuffle=False)
    
    edge_sage_train_loader = DataLoader([edge_sage_dataset[i] for i in train_indices], batch_size=batch_size, shuffle=True)
    edge_sage_val_loader = DataLoader([edge_sage_dataset[i] for i in val_indices], batch_size=batch_size, shuffle=False)
    edge_sage_test_loader = DataLoader([edge_sage_dataset[i] for i in test_indices], batch_size=batch_size, shuffle=False)
    
    # 创建模型 (使用从数据中获取的维度)
    print("创建模型...")
    if feature_dim is None or num_classes is None or num_classes == 0:
         print("错误：未能从数据中确定特征维度或类别数量，无法创建模型。")
         return

    baseline_model = BaselineSAGE(
        in_channels=feature_dim, # 使用实际特征维度
        hidden_channels=hidden_dim,
        out_channels=num_classes, # 使用实际类别数
        num_layers=2,
        dropout=0.1
    )
    
    rgcn_model = RGCN(
        in_channels=feature_dim, # 使用实际特征维度
        hidden_channels=hidden_dim,
        out_channels=num_classes, # 使用实际类别数
        num_relations=num_relations,
        num_layers=2,
        dropout=0.1,
        num_bases=4
    )
    
    edge_sage_model = EdgeCondSAGE(
        in_channels=feature_dim, # 使用实际特征维度
        hidden_channels=hidden_dim,
        out_channels=num_classes, # 使用实际类别数
        edge_dim=edge_dim,
        num_layers=2,
        dropout=0.1
    )
    
    # 创建训练器
    print("创建训练器...")
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
    
    # 训练模型
    print("训练基准模型...")
    baseline_history = baseline_trainer.train(
        train_loader=baseline_train_loader,
        val_loader=baseline_val_loader,
        num_epochs=num_epochs,
        metric_fn=accuracy,
        early_stopping=5,
        save_dir=os.path.join(output_dir, 'baseline')
    )
    
    print("训练R-GCN模型...")
    rgcn_history = rgcn_trainer.train(
        train_loader=rgcn_train_loader,
        val_loader=rgcn_val_loader,
        num_epochs=num_epochs,
        metric_fn=accuracy,
        early_stopping=5,
        save_dir=os.path.join(output_dir, 'rgcn')
    )
    
    print("训练Edge-SAGE模型...")
    edge_sage_history = edge_sage_trainer.train(
        train_loader=edge_sage_train_loader,
        val_loader=edge_sage_val_loader,
        num_epochs=num_epochs,
        metric_fn=accuracy,
        early_stopping=5,
        save_dir=os.path.join(output_dir, 'edge_sage')
    )
    
    # 创建评估器
    print("评估模型...")
    evaluator = ModelEvaluator(task_type='node')
    evaluator.add_model('Baseline', baseline_trainer)
    evaluator.add_model('RGCN', rgcn_trainer)
    evaluator.add_model('EdgeSAGE', edge_sage_trainer)
    
    # 可视化训练历史
    evaluator.visualize_training_history(save_path=os.path.join(output_dir, 'training_history.png'))
    
    # # 模拟Querier任务（图相似度比较） - 暂时注释掉，因为训练目标是节点分类
    # print("\n评估Querier任务（示例推荐）...")
    # querier_results = evaluator.evaluate_example_recommendation(baseline_test_loader) 
    
    # # 保存性能比较表格 - 暂时注释掉
    # performance_df = evaluator.compare_model_performance(task='example_recommendation')
    # print("\n性能比较:")
    # print(performance_df)
    
    # 导出结果
    evaluator.export_results(os.path.join(output_dir, 'evaluation'))
    
    # 打印参数数量比较
    print("\n模型参数量比较:")
    # for name, result in querier_results.items(): # 暂时注释掉
    #     print(f"{name}: {result['parameters']:,} 参数")
    # 改为直接从 trainer 获取参数量
    print(f"Baseline: {sum(p.numel() for p in baseline_model.parameters()):,} 参数")
    print(f"RGCN: {sum(p.numel() for p in rgcn_model.parameters()):,} 参数")
    print(f"EdgeSAGE: {sum(p.numel() for p in edge_sage_model.parameters()):,} 参数")

    
    # 为第一个测试图可视化节点嵌入
    if len(baseline_test_loader) > 0:
        # 注意：DataLoader 可能返回 batch 对象，我们需要从中提取一个图
        # 或者直接加载一个测试图数据
        test_data_indices = test_indices[:batch_size] # 取测试集的一个批次索引
        test_batch = baseline_test_loader.collate_fn([base_dataset[i] for i in test_data_indices])

        evaluator.visualize_node_embeddings(
            test_batch, # 传递批次对象
            save_path=os.path.join(output_dir, 'node_embeddings.png')
        )
    
    print(f"\n所有结果已保存到 {output_dir}")


if __name__ == "__main__":
    # 执行训练和评估
    train_and_evaluate_models() 