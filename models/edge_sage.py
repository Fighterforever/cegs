import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing


class EdgeCondSAGELayer(MessagePassing):
    """
    边条件聚合的GraphSAGE层实现
    """
    def __init__(self, in_channels, out_channels, edge_dim, aggr='mean', normalize=True, 
                 bias=True, **kwargs):
        """
        初始化边条件GraphSAGE层
        
        Args:
            in_channels (int): 输入特征维度
            out_channels (int): 输出特征维度
            edge_dim (int): 边特征维度
            aggr (str): 聚合函数类型 ('mean', 'max', 'sum', 'lstm')
            normalize (bool): 是否归一化
            bias (bool): 是否使用偏置
        """
        super(EdgeCondSAGELayer, self).__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.normalize = normalize
        
        # 消息生成 MLP，接收拼接的[节点特征, 边特征]
        self.message_mlp = nn.Linear(in_channels + edge_dim, out_channels)
        
        # 更新函数，接收拼接的[节点特征, 聚合邻居特征]
        self.update_mlp = nn.Linear(in_channels + out_channels, out_channels)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.message_mlp.weight)
        nn.init.zeros_(self.message_mlp.bias)
        nn.init.xavier_uniform_(self.update_mlp.weight)
        nn.init.zeros_(self.update_mlp.bias)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播
        
        Args:
            x (Tensor): 节点特征矩阵，形状 [num_nodes, in_channels]
            edge_index (LongTensor): 边索引，形状 [2, num_edges]
            edge_attr (Tensor): 边特征/角色，形状 [num_edges, edge_dim]
            
        Returns:
            Tensor: 更新后的节点特征，形状 [num_nodes, out_channels]
        """
        # 如果没有提供边特征，则使用全零边特征
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), 
                                   device=x.device)
        
        # 进行消息传递
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # 更新节点表示
        out = torch.cat([x, out], dim=1)
        out = self.update_mlp(out)
        
        # 归一化
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
            
        # 添加偏置
        if self.bias is not None:
            out = out + self.bias
            
        return out
    
    def message(self, x_j, edge_attr):
        """
        计算消息
        
        Args:
            x_j (Tensor): 源节点特征，形状 [num_edges, in_channels]
            edge_attr (Tensor): 边特征，形状 [num_edges, edge_dim]
            
        Returns:
            Tensor: 生成的消息，形状 [num_edges, out_channels]
        """
        # 将节点特征与边特征拼接
        message_input = torch.cat([x_j, edge_attr], dim=1)
        # 通过MLP生成消息
        return F.relu(self.message_mlp(message_input))


class EdgeCondSAGE(torch.nn.Module):
    """
    边条件聚合的GraphSAGE模型
    """
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, 
                 num_layers=2, dropout=0.0, aggr='mean', normalize=True, bias=True):
        """
        初始化边条件GraphSAGE模型
        
        Args:
            in_channels (int): 输入特征维度
            hidden_channels (int): 隐藏层维度
            out_channels (int): 输出特征维度
            edge_dim (int): 边特征维度
            num_layers (int): GNN层数
            dropout (float): Dropout概率
            aggr (str): 聚合函数类型 ('mean', 'max', 'sum', 'lstm')
            normalize (bool): 是否归一化
            bias (bool): 是否使用偏置
        """
        super(EdgeCondSAGE, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 创建边条件SAGE层
        self.convs = nn.ModuleList()
        
        # 第一层
        self.convs.append(EdgeCondSAGELayer(
            in_channels=in_channels,
            out_channels=hidden_channels,
            edge_dim=edge_dim,
            aggr=aggr,
            normalize=normalize,
            bias=bias
        ))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(EdgeCondSAGELayer(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                edge_dim=edge_dim,
                aggr=aggr,
                normalize=normalize,
                bias=bias
            ))
        
        # 最后一层
        if num_layers > 1:
            self.convs.append(EdgeCondSAGELayer(
                in_channels=hidden_channels,
                out_channels=out_channels,
                edge_dim=edge_dim,
                aggr=aggr,
                normalize=normalize,
                bias=bias
            ))
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播
        
        Args:
            x (Tensor): 节点特征矩阵，形状 [num_nodes, in_channels]
            edge_index (LongTensor): 边索引，形状 [2, num_edges]
            edge_attr (Tensor): 边特征/角色，形状 [num_edges, edge_dim]
            
        Returns:
            Tensor: 最终节点嵌入，形状 [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
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
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        return self.forward(data.x, data.edge_index, edge_attr) 