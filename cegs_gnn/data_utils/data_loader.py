import os
import torch
import numpy as np
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import from_networkx
import networkx as nx


class EdgeRoleEncoder:
    """
    边角色编码器，将类别化的边角色转换为模型输入格式
    """
    def __init__(self, role_vocab=None, embedding_dim=None):
        """
        初始化边角色编码器
        
        Args:
            role_vocab (list, optional): 预定义的边角色词汇表，默认为None（动态构建）
            embedding_dim (int, optional): 嵌入向量维度，用于EdgeCondSAGE，默认为None
        """
        self.role_vocab = role_vocab
        self.role_to_idx = {}
        self.idx_to_role = {}
        self.embedding_dim = embedding_dim
        self.embedding = None
        
        # 如果提供了预定义词汇表，则初始化映射
        if role_vocab is not None:
            self._build_mapping()
            
            # 如果指定了嵌入维度，创建嵌入层
            if embedding_dim is not None:
                self.embedding = torch.nn.Embedding(len(role_vocab), embedding_dim)
    
    def _build_mapping(self):
        self.role_to_idx = {role: i for i, role in enumerate(self.role_vocab)}
        self.idx_to_role = {i: role for i, role in enumerate(self.role_vocab)}
    
    def fit(self, role_list):
        """
        从数据中构建边角色词汇表
        
        Args:
            role_list (list): 所有边角色的列表
            
        Returns:
            self: 返回自身，便于链式调用
        """
        # 如果没有预定义词汇表，则从数据中构建
        if self.role_vocab is None:
            unique_roles = sorted(list(set(role_list)))
            self.role_vocab = unique_roles
            
            self._build_mapping()
            
            # 如果指定了嵌入维度，创建嵌入层
            if self.embedding_dim is not None:
                self.embedding = torch.nn.Embedding(len(unique_roles), self.embedding_dim)
                
        return self
    
    def transform_to_edge_type(self, roles):
        """
        将边角色转换为边类型索引（用于R-GCN）
        
        Args:
            roles (list): 边角色列表
            
        Returns:
            torch.LongTensor: 边类型索引
        """
        unknown_index = 0 
        indices = [self.role_to_idx.get(role, unknown_index) for role in roles]
        return torch.tensor(indices, dtype=torch.long)
    
    def transform_to_edge_attr(self, roles):
        """
        将边角色转换为边属性嵌入（用于EdgeCondSAGE）
        
        Args:
            roles (list): 边角色列表
            
        Returns:
            torch.Tensor: 边属性嵌入
        """
        if self.embedding is None:
            raise ValueError("边角色嵌入未初始化，请先设置embedding_dim并调用fit()方法")
            
        unknown_index = 0
        indices = [self.role_to_idx.get(role, unknown_index) for role in roles]
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        return self.embedding(indices_tensor)
    
    def get_num_relations(self):
        """
        获取关系类型数量（用于R-GCN）
        
        Returns:
            int: 关系类型数量
        """
        return len(self.role_vocab) if self.role_vocab else 0


class CEGSGraphDataset(InMemoryDataset):
    """
    CEGS图数据集，用于加载和预处理CEGS中的图数据
    """
    def __init__(self, root, graphs, node_features, edge_roles=None, node_labels=None, 
                 process_for_rgcn=False, process_for_edge_sage=False, edge_embedding_dim=None, 
                 transform=None, pre_transform=None):
        """
        初始化CEGS图数据集
        
        Args:
            root (str): 数据集根目录
            graphs (list): 图列表，每个图为NetworkX格式
            node_features (list): 节点特征列表，每个元素是一个dict，字典键为节点ID，值为特征向量
            edge_roles (list, optional): 边角色列表，每个元素是一个dict，字典键为边(u,v)，值为角色标签
            node_labels (list, optional): 节点标签列表，每个元素是一个dict，字典键为节点ID，值为标签
            process_for_rgcn (bool): 是否生成 edge_type
            process_for_edge_sage (bool): 是否生成 edge_attr
            edge_embedding_dim (int, optional): edge_attr 的嵌入维度
            transform (callable, optional): 数据变换函数
            pre_transform (callable, optional): 预处理变换函数
        """
        self.graphs = graphs
        self.node_features = node_features
        self.edge_roles_list = edge_roles
        self.node_labels_list = node_labels
        self.process_for_rgcn = process_for_rgcn
        self.process_for_edge_sage = process_for_edge_sage
        self.edge_embedding_dim = edge_embedding_dim
        self._encoder = None # 内部编码器实例
        
        # 确保数据大小一致
        assert len(graphs) == len(node_features), "图数量与节点特征数量不匹配"
        if self.edge_roles_list is not None:
            assert len(graphs) == len(self.edge_roles_list), "图数量与边角色数量不匹配"
        if self.node_labels_list is not None:
            assert len(graphs) == len(self.node_labels_list), "图数量与节点标签数量不匹配"
        
        super(CEGSGraphDataset, self).__init__(root, transform, pre_transform)
        # 加载处理后的数据
        self.data, self.slices = torch.load(self.processed_paths[0])
        # 如果需要，从加载的数据中恢复编码器状态 (仅在加载缓存时需要)
        if os.path.exists(self.processed_paths[0]) and (self.process_for_rgcn or self.process_for_edge_sage):
             self._load_encoder(root)

    @property
    def raw_file_names(self):
        # 虽然我们直接传入数据，但仍需定义，可以为空
        return [] 
    
    @property
    def processed_file_names(self):
        # 根据处理类型和版本号生成不同的缓存文件名
        version_suffix = "_v3" # 添加版本号
        type_suffix = ""
        if self.process_for_rgcn:
            type_suffix += "_rgcn"
        if self.process_for_edge_sage:
            type_suffix += f"_edgesage{self.edge_embedding_dim or ''}"
        return [f'processed_data{type_suffix}{version_suffix}.pt']

    def _save_encoder(self, root):
        if self._encoder is not None:
             os.makedirs(os.path.join(root, 'processed'), exist_ok=True)
             torch.save(self._encoder, os.path.join(root, 'processed', 'encoder.pt'))

    def _load_encoder(self, root):
        encoder_path = os.path.join(root, 'processed', 'encoder.pt')
        if os.path.exists(encoder_path):
             self._encoder = torch.load(encoder_path)

    def num_relations(self):
         # 提供访问关系数量的方法
         if self._encoder:
             return self._encoder.get_num_relations()
         return 0

    def process(self):
        """处理数据，将NetworkX图转换为PyG数据对象，并可选生成edge_type/edge_attr"""
        data_list = []

        # 如果需要处理边角色，先初始化并拟合编码器
        if (self.process_for_rgcn or self.process_for_edge_sage) and self.edge_roles_list is not None:
            all_roles = [role for graph_roles in self.edge_roles_list for role in graph_roles.values()]
            # 创建编码器实例
            self._encoder = EdgeRoleEncoder(embedding_dim=self.edge_embedding_dim if self.process_for_edge_sage else None)
            self._encoder.fit(all_roles)
            # 保存编码器状态
            self._save_encoder(self.root)

        for i, nx_graph in enumerate(self.graphs):
            # 获取节点特征
            node_feat_dict = self.node_features[i]
            
            # 确保图中的节点与特征字典中的节点一致
            for node in nx_graph.nodes():
                if node not in node_feat_dict:
                    raise ValueError(f"节点{node}在特征字典中不存在")
            
            # 创建节点特征张量，确保节点顺序一致
            nodes = sorted(nx_graph.nodes())
            x = torch.stack([torch.tensor(node_feat_dict[node]) for node in nodes])
            
            # 将NetworkX图转换为PyG数据对象 (只包含 x 和 edge_index)
            # 注意：直接用 from_networkx 可能包含原始属性，我们稍后只添加需要的
            edge_index = from_networkx(nx_graph).edge_index
            data = Data(x=x, edge_index=edge_index)
            
            # 处理节点标签（如果提供）
            if self.node_labels_list is not None:
                node_label_dict = self.node_labels_list[i]
                labels = []
                for node in nodes:
                    if node in node_label_dict:
                        labels.append(node_label_dict[node])
                    else:
                        raise ValueError(f"节点{node}在标签字典中不存在")
                data.y = torch.tensor(labels, dtype=torch.long)
            
            # 处理边角色（如果提供 且 需要）
            if self.edge_roles_list is not None and self._encoder is not None:
                 edge_role_dict = self.edge_roles_list[i]
                 edge_roles_for_data = []
                 # 需要确保边的顺序与 edge_index 对应
                 # from_networkx 返回的 edge_index 不保证顺序，我们需要自己构建
                 node_to_idx = {node: idx for idx, node in enumerate(nodes)}
                 map_edge_index_list = [[], []]
                 edge_roles_ordered = []

                 for u, v, role_data in nx_graph.edges(data=True): # 假设角色在 edge data 中，或者我们需要从 dict 查找
                      # 从字典查找角色
                      role = edge_role_dict.get((u, v), edge_role_dict.get((v, u), "unknown"))

                      # 获取节点索引
                      src, dst = node_to_idx.get(u), node_to_idx.get(v)
                      if src is None or dst is None: continue # 跳过不在 node_features 中的节点

                      # 添加边 (双向，因为 from_networkx 通常是无向的)
                      map_edge_index_list[0].extend([src, dst])
                      map_edge_index_list[1].extend([dst, src])
                      edge_roles_ordered.extend([role, role]) # 角色也对应加两次

                 # 更新 edge_index 和 roles
                 data.edge_index = torch.tensor(map_edge_index_list, dtype=torch.long)
                 edge_roles_for_data = edge_roles_ordered

                 # 根据需要生成 edge_type 或 edge_attr
                 if self.process_for_rgcn:
                     data.edge_type = self._encoder.transform_to_edge_type(edge_roles_for_data)
                 if self.process_for_edge_sage:
                     data.edge_attr = self._encoder.transform_to_edge_attr(edge_roles_for_data)
            
            # 应用预变换 (如果需要)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            data_list.append(data)
        
        # 检查 data_list 是否为空
        if not data_list:
             print(f"Warning: No data processed for dataset at root {self.root}")
             # 创建一个空的 data 对象以避免 collate 失败
             data = Data()
             slices = {key: torch.tensor([0, 0], dtype=torch.long) for key in data.keys}

        else:
             # 保存处理后的数据
             data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])


def load_synthetic_data(num_graphs=10, nodes_per_graph=10, edge_prob=0.3, 
                       feature_dim=16, num_edge_roles=5, num_classes=5, seed=42):
    """
    加载合成数据，用于测试
    
    Args:
        num_graphs (int): 生成的图数量
        nodes_per_graph (int): 每个图的节点数量
        edge_prob (float): 随机图中的边概率
        feature_dim (int): 节点特征维度
        num_edge_roles (int): 边角色数量
        num_classes (int): 节点类别数量
        seed (int): 随机种子
        
    Returns:
        tuple: (graphs, node_features, edge_roles, node_labels)，分别是NetworkX图列表、节点特征列表、边角色列表和节点标签列表
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    graphs = []
    node_features = []
    edge_roles = []
    node_labels = []
    
    # 生成边角色词汇表
    role_vocab = [f"role_{i}" for i in range(num_edge_roles)]
    
    for i in range(num_graphs):
        # 创建随机图
        g = nx.erdos_renyi_graph(nodes_per_graph, edge_prob, seed=seed+i)
        
        # 生成随机节点特征和标签
        features = {}
        labels = {}
        for node in g.nodes():
            features[node] = torch.randn(feature_dim)
            labels[node] = np.random.randint(0, num_classes)
        
        # 随机分配边角色
        roles = {}
        for u, v in g.edges():
            roles[(u, v)] = np.random.choice(role_vocab)
        
        graphs.append(g)
        node_features.append(features)
        edge_roles.append(roles)
        node_labels.append(labels)
    
    return graphs, node_features, edge_roles, node_labels


# --- 新的、更有意义的合成数据生成器 ---
NODE_TYPES = ['Start', 'Assignment', 'FunctionCall', 'IfStatement', 'Return', 'End']
EDGE_ROLES = ['next_statement', 'calls', 'condition_true', 'condition_false', 'returns_to']

def load_cegs_like_synthetic_data(num_graphs=10, min_nodes=8, max_nodes=20, 
                                  if_prob=0.2, call_prob=0.15, seed=42):
    """
    生成模拟代码结构的合成图数据 (CEGS-like)

    Args:
        num_graphs (int): 生成的图数量
        min_nodes (int): 每个图的最小节点数
        max_nodes (int): 每个图的最大节点数
        if_prob (float): 遇到 IfStatement 的概率
        call_prob (float): 遇到 FunctionCall 的概率
        seed (int): 随机种子

    Returns:
        tuple: (graphs, node_features, edge_roles, node_labels)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    graphs_nx = []
    node_features_list = []
    edge_roles_list = []
    node_labels_list = []

    node_type_to_idx = {name: i for i, name in enumerate(NODE_TYPES)}
    feature_dim = len(NODE_TYPES)
    num_classes = 3 # 0: Standard, 1: Control, 2: Action/Return
    num_edge_roles = len(EDGE_ROLES)

    for i in range(num_graphs):
        g = nx.DiGraph() # 使用有向图
        features = {}
        labels = {}
        roles = {}
        num_nodes = np.random.randint(min_nodes, max_nodes + 1)
        node_counter = 0

        def add_node(node_type):
            nonlocal node_counter
            node_id = node_counter
            g.add_node(node_id)
            
            # 特征: one-hot encoding
            feature_vec = torch.zeros(feature_dim)
            feature_vec[node_type_to_idx[node_type]] = 1
            features[node_id] = feature_vec + torch.randn(feature_dim) * 0.1 # 添加少量噪声
            
            # 标签: 根据类型分配
            if node_type in ['Start', 'Assignment', 'End']:
                labels[node_id] = 0 # Standard
            elif node_type == 'IfStatement':
                labels[node_id] = 1 # Control
            else: # FunctionCall, Return
                labels[node_id] = 2 # Action/Return
                
            node_counter += 1
            return node_id

        def add_edge(u, v, role_name):
            g.add_edge(u, v)
            roles[(u, v)] = role_name

        # 生成图结构
        current_node = add_node('Start')
        pending_branches = [] # 用于处理 If/Else 结束后的汇合点
        call_stack = [] # 简单模拟调用返回

        while node_counter < num_nodes:
            rand_val = np.random.rand()
            
            if rand_val < if_prob and node_counter < num_nodes - 3: # 确保有足够节点创建分支
                if_node = add_node('IfStatement')
                add_edge(current_node, if_node, 'next_statement')
                
                # True 分支
                true_branch_start = add_node('Assignment') # 分支至少有一个节点
                add_edge(if_node, true_branch_start, 'condition_true')
                # (可以递归或循环生成更长分支，这里简化)
                true_branch_end = true_branch_start 

                # False 分支
                false_branch_start = add_node('Assignment')
                add_edge(if_node, false_branch_start, 'condition_false')
                false_branch_end = false_branch_start

                # 记录汇合点
                pending_branches.append((true_branch_end, false_branch_end))
                current_node = if_node # 当前节点变为 if 节点，等待分支结束
                # 在这个简化版本中，我们不立即继续生成分支，而是让主循环在下一次迭代处理挂起的节点
                # 或者更合适的做法是让 current_node 指向其中一个分支的末端，这里暂时中断当前路径
                if pending_branches: # 切换到最早的未完成分支的末端
                     current_node = pending_branches[0][0] # 随便选一个分支继续
                else: # 如果没有分支了，就中断（或添加 End）
                    break
            
            elif rand_val < if_prob + call_prob and node_counter < num_nodes - 1:
                 call_node = add_node('FunctionCall')
                 add_edge(current_node, call_node, 'next_statement')
                 # 模拟调用（简化：不真正创建子图，只添加返回边）
                 return_node = add_node('Return')
                 # add_edge(call_node, ..., 'calls') # 指向被调用函数（省略）
                 add_edge(return_node, call_node, 'returns_to') 
                 current_node = call_node # 调用结束后，控制权返回 call_node
                 # (更真实的应该是返回 call_node 的下一个语句)
                 # 简化：直接从 call_node 继续

            else: # Assignment node
                assign_node = add_node('Assignment')
                add_edge(current_node, assign_node, 'next_statement')
                current_node = assign_node

            # 检查是否有分支需要汇合
            if pending_branches and current_node in [p[0] for p in pending_branches] + [p[1] for p in pending_branches]:
                 # 找到哪个分支结束了
                 finished_branch_info = None
                 for idx, (true_end, false_end) in enumerate(pending_branches):
                     if current_node == true_end or current_node == false_end:
                         # 简单处理：假设另一个分支也结束了（或等待它结束）
                         # 这里我们假设当一个分支到达其末端节点时，它就准备好汇合了
                         # 找到配对的分支末端
                         other_end = false_end if current_node == true_end else true_end
                         # 简化：直接汇合，创建汇合节点
                         if node_counter < num_nodes:
                            merge_node = add_node('Assignment') # 汇合后通常是普通语句
                            add_edge(current_node, merge_node, 'next_statement')
                            # 假设另一个分支也指向这个汇合点（或者需要更复杂的逻辑来找到真正的另一个分支末端）
                            if g.has_node(other_end):
                                 add_edge(other_end, merge_node, 'next_statement')
                            current_node = merge_node
                            finished_branch_info = idx
                         break # 处理完一个汇合
                 if finished_branch_info is not None:
                    pending_branches.pop(finished_branch_info)

        # 确保图有结束节点
        if not list(g.successors(current_node)): # 如果当前节点没有后继
            end_node = add_node('End')
            add_edge(current_node, end_node, 'next_statement')
        
        # 清理孤立节点 (以防万一)
        isolated_nodes = list(nx.isolates(g))
        g.remove_nodes_from(isolated_nodes)
        features = {k: v for k, v in features.items() if k not in isolated_nodes}
        labels = {k: v for k, v in labels.items() if k not in isolated_nodes}
        # (边角色字典不需要清理，因为边依赖于节点)
        if not g.nodes: continue # 如果图变空了，跳过

        graphs_nx.append(g)
        node_features_list.append(features)
        edge_roles_list.append(roles)
        node_labels_list.append(labels)

    print(f"Generated {len(graphs_nx)} CEGS-like graphs.")
    print(f"Feature dim: {feature_dim}, Num classes: {num_classes}, Num edge roles: {num_edge_roles}")
    return graphs_nx, node_features_list, edge_roles_list, node_labels_list


def prepare_data_for_models(root="./data", load_synthetic=True, use_cegs_like_synth=True, # 新增标志位
                             embedding_dim=8, 
                             # 参数现在由生成函数决定一部分，但训练脚本可能仍需提供
                             num_classes=None, feature_dim=None, num_relations=None):
    """
    准备数据，用于训练三种模型
    """
    # 加载数据
    if load_synthetic:
        if use_cegs_like_synth:
            print("Loading CEGS-like synthetic data...")
            graphs, node_features, edge_roles, node_labels = load_cegs_like_synthetic_data()
            # 从生成的数据获取维度信息
            if graphs: # 确保生成了数据
                 example_features = next(iter(node_features[0].values()))
                 feature_dim = example_features.shape[0]
                 all_labels = [lbl for graph_lbls in node_labels for lbl in graph_lbls.values()]
                 num_classes = len(set(all_labels)) if all_labels else 0 # 改为实际类别数
                 all_roles = [role for graph_roles in edge_roles for role in graph_roles.values()]
                 # num_relations 由 EdgeRoleEncoder 内部确定，后面从 dataset 获取
            else:
                 print("Warning: No CEGS-like graphs generated.")
                 feature_dim = feature_dim or 1 # 避免 None
                 num_classes = num_classes or 1

        else: # 使用旧的随机数据
            print("Loading original random synthetic data...")
            graphs, node_features, edge_roles, node_labels = load_synthetic_data(
                feature_dim=feature_dim or 16, num_classes=num_classes or 5 
            )
            # num_relations = 5 # 旧数据固定为5
    else:
        raise NotImplementedError("真实数据加载尚未实现")
    
    # 创建独立的数据集实例
    print(f"Creating datasets with feature_dim={feature_dim}, num_classes={num_classes}")
    base_dataset_root = os.path.join(root, 'cegs_like_base' if use_cegs_like_synth else 'base')
    rgcn_dataset_root = os.path.join(root, 'cegs_like_rgcn' if use_cegs_like_synth else 'rgcn')
    edgesage_dataset_root = os.path.join(root, f'cegs_like_edgesage_{embedding_dim}' if use_cegs_like_synth else f'edgesage_{embedding_dim}')

    # 清理旧缓存（如果更改了数据生成方式）
    # import shutil
    # if os.path.exists(os.path.join(base_dataset_root, 'processed')): shutil.rmtree(os.path.join(base_dataset_root, 'processed'))
    # if os.path.exists(os.path.join(rgcn_dataset_root, 'processed')): shutil.rmtree(os.path.join(rgcn_dataset_root, 'processed'))
    # if os.path.exists(os.path.join(edgesage_dataset_root, 'processed')): shutil.rmtree(os.path.join(edgesage_dataset_root, 'processed'))
    # print("Cleaned old dataset cache (if existed).")

    base_dataset = CEGSGraphDataset(
        root=base_dataset_root,
        graphs=graphs,
        node_features=node_features,
        edge_roles=edge_roles,
        node_labels=node_labels
    )
    
    rgcn_dataset = CEGSGraphDataset(
        root=rgcn_dataset_root, 
        graphs=graphs,
        node_features=node_features,
        edge_roles=edge_roles,
        node_labels=node_labels,
        process_for_rgcn=True 
    )
    
    edge_sage_dataset = CEGSGraphDataset(
        root=edgesage_dataset_root, 
        graphs=graphs,
        node_features=node_features,
        edge_roles=edge_roles,
        node_labels=node_labels,
        process_for_edge_sage=True, 
        edge_embedding_dim=embedding_dim
    )
    
    # 获取关系数量 (确保 rgcn_dataset 已处理)
    num_relations = rgcn_dataset.num_relations()
    print(f"Num relations detected: {num_relations}") 

    # 返回实际的类别数和特征维度给训练脚本
    return base_dataset, rgcn_dataset, edge_sage_dataset, num_relations, num_classes, feature_dim 