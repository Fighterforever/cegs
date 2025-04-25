import torch
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


def compute_graph_embedding(node_embeddings, pooling='mean'):
    """
    计算图嵌入
    
    Args:
        node_embeddings (torch.Tensor): 节点嵌入矩阵，形状为 [num_nodes, embedding_dim]
        pooling (str): 池化方法，可选 'mean', 'sum', 'max'
        
    Returns:
        torch.Tensor: 图嵌入向量，形状为 [embedding_dim]
    """
    if pooling == 'mean':
        return torch.mean(node_embeddings, dim=0)
    elif pooling == 'sum':
        return torch.sum(node_embeddings, dim=0)
    elif pooling == 'max':
        return torch.max(node_embeddings, dim=0)[0]
    else:
        raise ValueError(f"不支持的池化方法: {pooling}")


def compute_graph_similarity(graph_embeddings_1, graph_embeddings_2, method='cosine'):
    """
    计算图之间的相似度
    
    Args:
        graph_embeddings_1 (torch.Tensor): 第一组图嵌入，形状为 [num_graphs_1, embedding_dim]
        graph_embeddings_2 (torch.Tensor): 第二组图嵌入，形状为 [num_graphs_2, embedding_dim]
        method (str): 相似度度量方法，可选 'cosine', 'euclidean', 'dot'
        
    Returns:
        torch.Tensor: 相似度矩阵，形状为 [num_graphs_1, num_graphs_2]
    """
    # 转换为NumPy数组进行计算
    embeddings_1 = graph_embeddings_1.detach().cpu().numpy()
    embeddings_2 = graph_embeddings_2.detach().cpu().numpy()
    
    if method == 'cosine':
        # 计算余弦相似度
        similarity_matrix = cosine_similarity(embeddings_1, embeddings_2)
        return torch.tensor(similarity_matrix, device=graph_embeddings_1.device)
    
    elif method == 'euclidean':
        # 计算欧氏距离并转换为相似度（距离越小相似度越高）
        similarity_matrix = -np.sqrt(((embeddings_1[:, np.newaxis, :] - 
                                      embeddings_2[np.newaxis, :, :]) ** 2).sum(axis=2))
        return torch.tensor(similarity_matrix, device=graph_embeddings_1.device)
    
    elif method == 'dot':
        # 计算点积
        similarity_matrix = np.dot(embeddings_1, embeddings_2.T)
        return torch.tensor(similarity_matrix, device=graph_embeddings_1.device)
    
    else:
        raise ValueError(f"不支持的相似度度量方法: {method}")


def compute_node_similarity(node_embeddings_1, node_embeddings_2, method='cosine'):
    """
    计算节点之间的相似度
    
    Args:
        node_embeddings_1 (torch.Tensor): 第一组节点嵌入，形状为 [num_nodes_1, embedding_dim]
        node_embeddings_2 (torch.Tensor): 第二组节点嵌入，形状为 [num_nodes_2, embedding_dim]
        method (str): 相似度度量方法，可选 'cosine', 'euclidean', 'dot'
        
    Returns:
        torch.Tensor: 相似度矩阵，形状为 [num_nodes_1, num_nodes_2]
    """
    return compute_graph_similarity(node_embeddings_1, node_embeddings_2, method)


def find_node_mapping(similarity_matrix, method='greedy'):
    """
    根据节点相似度矩阵找到最佳节点映射
    
    Args:
        similarity_matrix (torch.Tensor): 节点相似度矩阵，形状为 [num_nodes_1, num_nodes_2]
        method (str): 映射方法，可选 'greedy', 'hungarian'
        
    Returns:
        list of tuple: 节点映射关系，每个元素为 (node_idx_1, node_idx_2)
    """
    if method == 'greedy':
        # 使用贪心算法
        # 每次选择相似度最高的节点对
        sim_matrix = similarity_matrix.detach().cpu().numpy()
        num_nodes_1, num_nodes_2 = sim_matrix.shape
        
        mappings = []
        available_nodes_1 = set(range(num_nodes_1))
        available_nodes_2 = set(range(num_nodes_2))
        
        while available_nodes_1 and available_nodes_2:
            # 找到最大相似度的节点对
            max_sim = -float('inf')
            best_pair = None
            
            for i in available_nodes_1:
                for j in available_nodes_2:
                    if sim_matrix[i, j] > max_sim:
                        max_sim = sim_matrix[i, j]
                        best_pair = (i, j)
            
            if best_pair is None:
                break
                
            # 添加匹配并移除已匹配的节点
            mappings.append(best_pair)
            available_nodes_1.remove(best_pair[0])
            available_nodes_2.remove(best_pair[1])
            
        return mappings
    
    elif method == 'hungarian':
        # 使用匈牙利算法（最优分配问题）
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            raise ImportError("请安装scipy以使用匈牙利算法: pip install scipy")
            
        # 将相似度转换为成本（最大化相似度等价于最小化负相似度）
        cost_matrix = -similarity_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        return list(zip(row_ind, col_ind))
    
    else:
        raise ValueError(f"不支持的映射方法: {method}")


def visualize_graph(nx_graph, node_roles=None, edge_roles=None, 
                    node_embeddings=None, save_path=None):
    """
    可视化图
    
    Args:
        nx_graph (networkx.Graph): NetworkX图对象
        node_roles (dict, optional): 节点角色字典，键为节点ID，值为角色
        edge_roles (dict, optional): 边角色字典，键为边(u,v)，值为角色
        node_embeddings (dict, optional): 节点嵌入字典，键为节点ID，值为嵌入向量
        save_path (str, optional): 图像保存路径
    """
    # 由于这是一个可视化函数，为简化起见，这里只提供代码框架
    # 在实际应用中，您需要实现完整的可视化逻辑
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("请安装matplotlib以使用可视化功能: pip install matplotlib")
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 设置节点颜色和标签
    node_colors = []
    node_labels = {}
    
    if node_roles:
        # 根据节点角色设置颜色
        role_to_color = {}  # 映射角色到颜色
        for node, role in node_roles.items():
            if role not in role_to_color:
                role_to_color[role] = len(role_to_color) / len(set(node_roles.values()))
            node_colors.append(role_to_color[role])
            node_labels[node] = f"{node}:{role}"
    else:
        node_colors = 'skyblue'
        node_labels = {node: str(node) for node in nx_graph.nodes()}
    
    # 设置边颜色和标签
    edge_colors = []
    edge_labels = {}
    
    if edge_roles:
        # 根据边角色设置颜色
        role_to_color = {}  # 映射角色到颜色
        for (u, v), role in edge_roles.items():
            if role not in role_to_color:
                role_to_color[role] = len(role_to_color) / len(set(edge_roles.values()))
            edge_colors.append(role_to_color[role])
            edge_labels[(u, v)] = role
    else:
        edge_colors = 'gray'
        
    # 使用spring布局
    pos = nx.spring_layout(nx_graph, seed=42)
    
    # 绘制图
    nx.draw_networkx_nodes(nx_graph, pos, node_color=node_colors, node_size=500, alpha=0.8)
    nx.draw_networkx_edges(nx_graph, pos, edge_color=edge_colors, width=2, alpha=0.5)
    nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=10)
    
    if edge_labels:
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=8)
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def generate_graph_batches(data_list, batch_size, shuffle=True):
    """
    生成图批次数据
    
    Args:
        data_list (list): PyG数据对象列表
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        
    Yields:
        list: 批次数据，包含batch_size个PyG数据对象
    """
    # 获取数据索引
    indices = list(range(len(data_list)))
    
    # 如果需要，打乱数据
    if shuffle:
        np.random.shuffle(indices)
    
    # 生成批次
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        yield [data_list[idx] for idx in batch_indices] 