import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing


class RGCNLayer(MessagePassing):
    """
    关系图卷积网络(R-GCN)层实现
    """
    def __init__(self, in_channels, out_channels, num_relations, 
                 num_bases=None, num_blocks=None, bias=True, **kwargs):
        """
        初始化R-GCN层
        
        Args:
            in_channels (int): 输入特征维度
            out_channels (int): 输出特征维度
            num_relations (int): 关系类型数量
            num_bases (int, optional): 基分解中的基数量，默认为None（不使用基分解）
            num_blocks (int, optional): 块对角分解中的块数量，默认为None（不使用块对角分解）
            bias (bool): 是否使用偏置
        """
        super(RGCNLayer, self).__init__(aggr='add', **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        
        # 根据指定的参数共享策略创建权重矩阵
        if num_bases is not None:
            # 基分解
            self.weight = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
            self.comp = nn.Parameter(torch.Tensor(num_relations, num_bases))
            self.use_basis = True
            self.use_block = False
        elif num_blocks is not None:
            # 块对角分解
            assert (in_channels % num_blocks == 0) and (out_channels % num_blocks == 0), \
                "特征维度必须能被块数整除"
            self.block_in_size = in_channels // num_blocks
            self.block_out_size = out_channels // num_blocks
            self.weight = nn.Parameter(
                torch.Tensor(num_relations, num_blocks, self.block_in_size, self.block_out_size)
            )
            self.use_basis = False
            self.use_block = True
        else:
            # 为每种关系使用独立的权重矩阵
            self.weight = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
            self.use_basis = False
            self.use_block = False
            
        # 自环权重
        self.self_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        if self.use_basis:
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.comp)
        elif self.use_block:
            nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.xavier_uniform_(self.weight)
        
        nn.init.xavier_uniform_(self.self_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_type=None):
        """
        前向传播
        
        Args:
            x (Tensor): 节点特征矩阵，形状 [num_nodes, in_channels]
            edge_index (LongTensor): 边索引，形状 [2, num_edges]
            edge_type (LongTensor): 边类型/关系，形状 [num_edges]
            
        Returns:
            Tensor: 更新后的节点特征，形状 [num_nodes, out_channels]
        """
        # 处理自环（自身信息）
        self_loop = torch.matmul(x, self.self_weight)
        
        # 如果没有提供边类型，则假定所有边都属于关系类型0
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
            
        # 进行消息传递
        out = self.propagate(edge_index, x=x, edge_type=edge_type)
        
        # 组合自环和邻居信息
        out = out + self_loop
        
        # 添加偏置
        if self.bias is not None:
            out = out + self.bias
            
        return out
    
    def message(self, x_j, edge_type):
        """
        计算消息
        
        Args:
            x_j (Tensor): 源节点特征，形状 [num_edges, in_channels]
            edge_type (Tensor): 边类型，形状 [num_edges]
            
        Returns:
            Tensor: 消息，形状 [num_edges, out_channels]
        """
        weight = self.compute_weight(edge_type)
        return torch.bmm(x_j.unsqueeze(1), weight).squeeze(1)
    
    def compute_weight(self, edge_type):
        """
        根据边类型计算对应的权重矩阵
        
        Args:
            edge_type (Tensor): 边类型，形状 [num_edges]
            
        Returns:
            Tensor: 每条边对应的权重矩阵，形状 [num_edges, in_channels, out_channels]
        """
        if self.use_basis:
            # 基分解: W_r = \sum_b a_{rb} * B_b
            weight = torch.einsum('rb,bij->rij', self.comp, self.weight)
        elif self.use_block:
            # 块对角分解
            weight = self.weight
            # 构建完整的权重矩阵
            weight_full = torch.zeros(self.num_relations, self.in_channels, self.out_channels, 
                                      device=weight.device)
            
            for r in range(self.num_relations):
                for b in range(self.num_blocks):
                    start_in, start_out = b * self.block_in_size, b * self.block_out_size
                    end_in, end_out = start_in + self.block_in_size, start_out + self.block_out_size
                    weight_full[r, start_in:end_in, start_out:end_out] = weight[r, b]
                    
            weight = weight_full
        else:
            # 直接使用为每种关系定义的权重矩阵
            weight = self.weight
            
        # 选择对应边类型的权重
        return weight[edge_type]


class RGCN(torch.nn.Module):
    """
    关系图卷积网络(R-GCN)模型
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, 
                 num_layers=2, dropout=0.0, num_bases=None, num_blocks=None, bias=True):
        """
        初始化R-GCN模型
        
        Args:
            in_channels (int): 输入特征维度
            hidden_channels (int): 隐藏层维度
            out_channels (int): 输出特征维度
            num_relations (int): 关系类型数量
            num_layers (int): GNN层数
            dropout (float): Dropout概率
            num_bases (int, optional): 基分解中的基数量，默认为None
            num_blocks (int, optional): 块对角分解中的块数量，默认为None
            bias (bool): 是否使用偏置
        """
        super(RGCN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 创建R-GCN层
        self.convs = nn.ModuleList()
        
        # 第一层
        self.convs.append(RGCNLayer(
            in_channels=in_channels,
            out_channels=hidden_channels,
            num_relations=num_relations,
            num_bases=num_bases,
            num_blocks=num_blocks,
            bias=bias
        ))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(RGCNLayer(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                num_relations=num_relations,
                num_bases=num_bases,
                num_blocks=num_blocks,
                bias=bias
            ))
        
        # 最后一层
        if num_layers > 1:
            self.convs.append(RGCNLayer(
                in_channels=hidden_channels,
                out_channels=out_channels,
                num_relations=num_relations,
                num_bases=num_bases,
                num_blocks=num_blocks,
                bias=bias
            ))
    
    def forward(self, x, edge_index, edge_type=None):
        """
        前向传播
        
        Args:
            x (Tensor): 节点特征矩阵，形状 [num_nodes, in_channels]
            edge_index (LongTensor): 边索引，形状 [2, num_edges]
            edge_type (LongTensor): 边类型/关系，形状 [num_edges]
            
        Returns:
            Tensor: 最终节点嵌入，形状 [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < self.num_layers - 1:  # 非最后一层应用激活函数和dropout
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def get_embeddings(self, data):
        """
        获取图或节点嵌入用于下游任务
        
        Args:
            data (Data): PyG数据对象
            
        Returns:
            Tensor: 节点嵌入，形状 [num_nodes, out_channels]
        """
        edge_type = data.edge_type if hasattr(data, 'edge_type') else None
        return self.forward(data.x, data.edge_index, edge_type) 