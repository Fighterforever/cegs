import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.models import GraphSAGE


class BaselineSAGELayer(torch.nn.Module):
    """
    基准GraphSAGE层实现
    """
    def __init__(self, in_channels, out_channels, aggr='mean', normalize=True, 
                 bias=True, **kwargs):
        super(BaselineSAGELayer, self).__init__()
        self.sage_conv = SAGEConv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr=aggr,
            normalize=normalize,
            bias=bias,
            **kwargs
        )
    
    def forward(self, x, edge_index, edge_attr=None, edge_type=None):
        """
        前向传播
        
        Args:
            x (Tensor): 节点特征矩阵，形状 [num_nodes, in_channels]
            edge_index (LongTensor): 边索引，形状 [2, num_edges]
            edge_attr (Tensor, optional): 边属性，在基准模型中未使用
            edge_type (LongTensor, optional): 边类型，在基准模型中未使用
            
        Returns:
            Tensor: 更新后的节点特征，形状 [num_nodes, out_channels]
        """
        # 基准模型忽略边属性和边类型
        return self.sage_conv(x, edge_index)


class BaselineSAGE(torch.nn.Module):
    """
    基准GraphSAGE模型
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 dropout=0.0, aggr='mean', normalize=True, bias=True):
        """
        初始化基准GraphSAGE模型
        
        Args:
            in_channels (int): 输入特征维度
            hidden_channels (int): 隐藏层维度
            out_channels (int): 输出特征维度
            num_layers (int): GNN层数
            dropout (float): Dropout概率
            aggr (str): 聚合函数类型 ('mean', 'max', 'sum', 'lstm')
            normalize (bool): 是否归一化
            bias (bool): 是否使用偏置
        """
        super(BaselineSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 创建GraphSAGE层
        self.convs = nn.ModuleList()
        
        # 第一层
        self.convs.append(BaselineSAGELayer(
            in_channels=in_channels,
            out_channels=hidden_channels,
            aggr=aggr,
            normalize=normalize,
            bias=bias
        ))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(BaselineSAGELayer(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                aggr=aggr,
                normalize=normalize,
                bias=bias
            ))
        
        # 最后一层
        if num_layers > 1:
            self.convs.append(BaselineSAGELayer(
                in_channels=hidden_channels,
                out_channels=out_channels,
                aggr=aggr,
                normalize=normalize,
                bias=bias
            ))
    
    def forward(self, x, edge_index, edge_attr=None, edge_type=None):
        """
        前向传播
        
        Args:
            x (Tensor): 节点特征矩阵，形状 [num_nodes, in_channels]
            edge_index (LongTensor): 边索引，形状 [2, num_edges]
            edge_attr (Tensor, optional): 边属性，在基准模型中未使用
            edge_type (LongTensor, optional): 边类型，在基准模型中未使用
            
        Returns:
            Tensor: 最终节点嵌入，形状 [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
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
        return self.forward(data.x, data.edge_index) 