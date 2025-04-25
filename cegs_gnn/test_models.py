import torch
from models.baseline import BaselineSAGE
from models.rgcn import RGCN
from models.edge_sage import EdgeCondSAGE
from data_utils.data_loader import load_synthetic_data

def test_models():
    print("测试CEGS GNN模型")
    
    # 参数设置
    node_dim = 16    # 节点特征维度
    edge_dim = 8     # 边特征维度
    hidden_dim = 64  # 隐藏层维度
    out_dim = 32     # 输出维度
    num_relations = 5 # 关系类型数量
    
    # 生成小型测试数据
    print("\n生成测试数据...")
    graphs, node_features, edge_roles = load_synthetic_data(
        num_graphs=2, 
        nodes_per_graph=5,
        feature_dim=node_dim,
        num_edge_roles=num_relations
    )
    
    # 手动构建一个简单的PyG格式输入
    print("\n构建测试输入...")
    num_nodes = 5
    x = torch.randn(num_nodes, node_dim)  # 节点特征
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                               [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)  # 边连接
    edge_type = torch.randint(0, num_relations, (edge_index.size(1),))  # 边类型
    edge_attr = torch.randn(edge_index.size(1), edge_dim)  # 边特征
    
    print(f"- 节点特征形状: {x.shape}")
    print(f"- 边索引形状: {edge_index.shape}")
    print(f"- 边类型形状: {edge_type.shape}")
    print(f"- 边特征形状: {edge_attr.shape}")
    
    # 测试基准GraphSAGE模型
    print("\n测试基准GraphSAGE模型...")
    baseline_model = BaselineSAGE(
        in_channels=node_dim,
        hidden_channels=hidden_dim,
        out_channels=out_dim,
        num_layers=2
    )
    baseline_out = baseline_model(x, edge_index)
    print(f"- 输出形状: {baseline_out.shape}")
    
    # 测试R-GCN模型
    print("\n测试R-GCN模型...")
    rgcn_model = RGCN(
        in_channels=node_dim,
        hidden_channels=hidden_dim,
        out_channels=out_dim,
        num_relations=num_relations,
        num_layers=2,
        num_bases=4  # 使用基分解
    )
    rgcn_out = rgcn_model(x, edge_index, edge_type)
    print(f"- 输出形状: {rgcn_out.shape}")
    
    # 测试带边条件的GraphSAGE模型
    print("\n测试EdgeCondSAGE模型...")
    edge_sage_model = EdgeCondSAGE(
        in_channels=node_dim,
        hidden_channels=hidden_dim,
        out_channels=out_dim,
        edge_dim=edge_dim,
        num_layers=2
    )
    edge_sage_out = edge_sage_model(x, edge_index, edge_attr)
    print(f"- 输出形状: {edge_sage_out.shape}")
    
    print("\n所有模型测试通过!")

if __name__ == "__main__":
    test_models() 