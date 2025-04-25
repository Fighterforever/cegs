import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def accuracy(y_pred, y_true):
    """
    计算分类准确率
    
    Args:
        y_pred (torch.Tensor): 预测值，形状为 [batch_size, num_classes] 或 [batch_size]
        y_true (torch.Tensor): 真实值，形状为 [batch_size]
        
    Returns:
        float: 准确率
    """
    if y_pred.dim() > 1:
        # 多分类问题，取最大值的索引作为预测类别
        y_pred = y_pred.argmax(dim=1)
    
    # 转换为NumPy数组
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    
    return accuracy_score(y_true, y_pred)


def precision(y_pred, y_true, average='macro'):
    """
    计算精确率
    
    Args:
        y_pred (torch.Tensor): 预测值，形状为 [batch_size, num_classes] 或 [batch_size]
        y_true (torch.Tensor): 真实值，形状为 [batch_size]
        average (str): 平均方式，可选 'micro', 'macro', 'weighted', 'samples'
        
    Returns:
        float: 精确率
    """
    if y_pred.dim() > 1:
        # 多分类问题，取最大值的索引作为预测类别
        y_pred = y_pred.argmax(dim=1)
    
    # 转换为NumPy数组
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def recall(y_pred, y_true, average='macro'):
    """
    计算召回率
    
    Args:
        y_pred (torch.Tensor): 预测值，形状为 [batch_size, num_classes] 或 [batch_size]
        y_true (torch.Tensor): 真实值，形状为 [batch_size]
        average (str): 平均方式，可选 'micro', 'macro', 'weighted', 'samples'
        
    Returns:
        float: 召回率
    """
    if y_pred.dim() > 1:
        # 多分类问题，取最大值的索引作为预测类别
        y_pred = y_pred.argmax(dim=1)
    
    # 转换为NumPy数组
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def f1(y_pred, y_true, average='macro'):
    """
    计算F1分数
    
    Args:
        y_pred (torch.Tensor): 预测值，形状为 [batch_size, num_classes] 或 [batch_size]
        y_true (torch.Tensor): 真实值，形状为 [batch_size]
        average (str): 平均方式，可选 'micro', 'macro', 'weighted', 'samples'
        
    Returns:
        float: F1分数
    """
    if y_pred.dim() > 1:
        # 多分类问题，取最大值的索引作为预测类别
        y_pred = y_pred.argmax(dim=1)
    
    # 转换为NumPy数组
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def mse(y_pred, y_true):
    """
    计算均方误差
    
    Args:
        y_pred (torch.Tensor): 预测值，形状为 [batch_size, ...]
        y_true (torch.Tensor): 真实值，形状为 [batch_size, ...]
        
    Returns:
        float: 均方误差
    """
    # 转换为NumPy数组
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    
    return mean_squared_error(y_true, y_pred)


def mae(y_pred, y_true):
    """
    计算平均绝对误差
    
    Args:
        y_pred (torch.Tensor): 预测值，形状为 [batch_size, ...]
        y_true (torch.Tensor): 真实值，形状为 [batch_size, ...]
        
    Returns:
        float: 平均绝对误差
    """
    # 转换为NumPy数组
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    
    return mean_absolute_error(y_true, y_pred)


def r2(y_pred, y_true):
    """
    计算R2分数
    
    Args:
        y_pred (torch.Tensor): 预测值，形状为 [batch_size, ...]
        y_true (torch.Tensor): 真实值，形状为 [batch_size, ...]
        
    Returns:
        float: R2分数
    """
    # 转换为NumPy数组
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    
    return r2_score(y_true, y_pred)


def top_k_accuracy(y_pred, y_true, k=1):
    """
    计算Top-K准确率
    
    Args:
        y_pred (torch.Tensor): 预测值，形状为 [batch_size, num_classes]
        y_true (torch.Tensor): 真实值，形状为 [batch_size]
        k (int): Top-K值
        
    Returns:
        float: Top-K准确率
    """
    # 确保预测值是二维的（批次大小 x 类别数）
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(1)
    
    # 获取Top-K预测
    _, topk_indices = torch.topk(y_pred, k, dim=1)
    
    # 检查真实值是否在Top-K预测中
    correct = torch.any(topk_indices == y_true.unsqueeze(1), dim=1)
    
    # 计算准确率
    return torch.mean(correct.float()).item()


def mean_reciprocal_rank(y_pred, y_true):
    """
    计算平均倒数排名（MRR）
    
    Args:
        y_pred (torch.Tensor): 预测值，形状为 [batch_size, num_classes]
        y_true (torch.Tensor): 真实值，形状为 [batch_size]
        
    Returns:
        float: MRR
    """
    # 确保预测值是二维的（批次大小 x 类别数）
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(1)
    
    # 对预测分数进行降序排序
    _, ranks = torch.sort(y_pred, dim=1, descending=True)
    
    # 获取每个样本真实类别的排名（找到真实类别在排序后的位置）
    # 注意：PyTorch的排名是从0开始的，所以需要加1
    batch_size = y_true.size(0)
    mrr = 0.0
    
    for i in range(batch_size):
        # 找到真实类别的索引
        true_class = y_true[i].item()
        
        # 找到该类别在排序后的位置
        rank_idx = torch.nonzero(ranks[i] == true_class, as_tuple=True)[0].item()
        
        # 计算倒数排名并累加
        mrr += 1.0 / (rank_idx + 1)
    
    # 计算平均值
    return mrr / batch_size


def graph_similarity_accuracy(similarity_matrix, target_indices):
    """
    计算图相似度准确率
    
    Args:
        similarity_matrix (torch.Tensor): 相似度矩阵，形状为 [num_query_graphs, num_candidate_graphs]
        target_indices (torch.Tensor): 目标图的索引，形状为 [num_query_graphs]
        
    Returns:
        float: 准确率
    """
    # 获取每个查询图最相似的候选图索引
    pred_indices = torch.argmax(similarity_matrix, dim=1)
    
    # 计算准确率
    correct = (pred_indices == target_indices).float()
    accuracy = torch.mean(correct).item()
    
    return accuracy


def node_mapping_accuracy(pred_mapping, true_mapping):
    """
    计算节点映射准确率
    
    Args:
        pred_mapping (list): 预测的节点映射，每个元素为 (source_idx, target_idx)
        true_mapping (list): 真实的节点映射，每个元素为 (source_idx, target_idx)
        
    Returns:
        float: 准确率
    """
    # 将映射转换为字典以便比较
    pred_dict = {src: tgt for src, tgt in pred_mapping}
    true_dict = {src: tgt for src, tgt in true_mapping}
    
    # 计算正确映射的数量
    correct = 0
    for src, tgt in true_dict.items():
        if src in pred_dict and pred_dict[src] == tgt:
            correct += 1
    
    # 计算准确率
    accuracy = correct / len(true_dict) if true_dict else 0.0
    
    return accuracy


def count_parameters(model):
    """
    计算模型参数量
    
    Args:
        model (torch.nn.Module): PyTorch模型
        
    Returns:
        int: 参数总量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 