import torch
import os
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# 从你的项目中导入数据集类和类型定义
from cegs_gnn.data_utils.data_loader import CEGSGraphDataset, NODE_TYPES, EDGE_ROLES

# 指定包含处理后数据的根目录
# root_dir = os.path.join('data', 'cegs_like_base') # 查看 base 数据
root_dir = os.path.join('data', 'cegs_like_rgcn') # 或者查看 rgcn 数据
# root_dir = os.path.join('data', 'cegs_like_edgesage_8') # 或者查看 edgesage 数据

processed_file_name = None
if 'rgcn' in root_dir:
    processed_file_name = 'processed_data_rgcn_v3.pt'
elif 'edgesage' in root_dir:
    # 从目录名提取 embedding_dim (更健壮的方式)
    try:
        embedding_dim = int(root_dir.split('_')[-1])
    except:
        embedding_dim = 8 # 回退默认值
    processed_file_name = f'processed_data_edgesage{embedding_dim}_v3.pt' 
else:
    processed_file_name = 'processed_data_v3.pt'

processed_file_path = os.path.join(root_dir, 'processed', processed_file_name)

if os.path.exists(processed_file_path):
    print(f"Loading dataset from cache: {root_dir}")
    # 实例化数据集，它会自动加载缓存
    # 注意：我们不需要传递原始数据（graphs, node_features 等），因为它们会从缓存加载
    # 但我们需要传递正确的标志位，以便它找到正确的缓存文件
    is_rgcn = 'rgcn' in root_dir
    is_edgesage = 'edgesage' in root_dir
    embedding_dim = 8 if is_edgesage else None 

    dataset = CEGSGraphDataset(
        root=root_dir, 
        graphs=[], node_features=[], # 传递空列表，因为会加载缓存
        process_for_rgcn=is_rgcn,
        process_for_edge_sage=is_edgesage,
        edge_embedding_dim=embedding_dim
        )
    
    print(f"Dataset loaded. Number of graphs: {len(dataset)}")

    if len(dataset) > 0:
        # 使用索引获取第一个图
        first_graph = dataset[0] 
        print("\nFirst graph in the dataset:")
        print(first_graph)
        print(f"  Number of nodes: {first_graph.num_nodes}")
        print(f"  Number of edges: {first_graph.num_edges}")
        print(f"  Features (x) shape: {first_graph.x.shape}")
        node_labels_available = hasattr(first_graph, 'y')
        edge_types_available = hasattr(first_graph, 'edge_type')
        edge_attrs_available = hasattr(first_graph, 'edge_attr') and first_graph.edge_attr is not None

        if node_labels_available:
            print(f"  Labels (y): {first_graph.y}")
            print(f"  Unique Labels: {torch.unique(first_graph.y)}")
        if edge_types_available:
            print(f"  Edge Type: {first_graph.edge_type}")
            print(f"  Unique Edge Types: {torch.unique(first_graph.edge_type)}")
        if edge_attrs_available:
            print(f"  Edge Attr shape: {first_graph.edge_attr.shape}")

        # --- 开始绘图 --- 
        print("\nDrawing the graph...")
        plt.figure(figsize=(12, 10))

        # 定义节点类别名称 (根据 data_loader 中的定义)
        node_class_names = {0: 'Standard', 1: 'Control', 2: 'Action/Return'}
        node_colors = []
        node_label_dict = {}
        if node_labels_available:
            labels = first_graph.y.cpu().numpy()
            num_classes = len(node_class_names)
            color_map = plt.cm.get_cmap('viridis', num_classes) 
            node_colors = [color_map(labels[i] / num_classes) for i in range(first_graph.num_nodes)]
            node_label_dict = {i: f"{i}\n({node_class_names.get(labels[i], 'Unknown')})" for i in range(first_graph.num_nodes)}
        else:
            node_colors = 'skyblue' # 默认颜色
            node_label_dict = {i: str(i) for i in range(first_graph.num_nodes)}

        # 转换回 NetworkX 图
        # 注意：对于有向边，to_networkx 默认可能只保留一种表示
        # 如果需要精确可视化，可能需要手动构建或调整参数
        graph_attributes = []
        if node_labels_available: graph_attributes.append('y')
        # 暂时不传递特征 'x' 以简化可视化
        edge_graph_attributes = []
        if edge_types_available: edge_graph_attributes.append('edge_type')
        # edge_attr 通常是高维嵌入，不适合直接可视化
        
        G = to_networkx(first_graph, node_attrs=graph_attributes, edge_attrs=edge_graph_attributes, to_undirected=False) # 保持有向

        # 获取布局
        pos = nx.spring_layout(G, seed=42, k=0.8) # 调整 k 可以改变节点间距

        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.8)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray', node_size=1000)

        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, labels=node_label_dict, font_size=8)

        # 绘制边标签 (如果 edge_type 可用)
        if edge_types_available and G.number_of_edges() > 0:
            try:
                edge_labels_raw = nx.get_edge_attributes(G, 'edge_type')
                # 将 edge_type 索引映射回名称
                edge_label_names = {edge: EDGE_ROLES[label_idx] 
                                    for edge, label_idx in edge_labels_raw.items() 
                                    if label_idx < len(EDGE_ROLES)} # 添加边界检查
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label_names, font_size=7, label_pos=0.3)
            except Exception as e:
                print(f"无法绘制边标签: {e}") # 可能是 to_networkx 未正确传递属性

        plt.title(f"Visualization of Graph 0 from {os.path.basename(root_dir)}")
        plt.axis('off') # 关闭坐标轴
        plt.show()
        # --- 绘图结束 --- 

    else:
        print("Dataset is empty.")

else:
    print(f"Processed file not found: {processed_file_path}")
    print("Please run the training script first to generate the data cache.")
