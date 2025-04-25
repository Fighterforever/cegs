import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import inspect

from ..data_utils.graph_utils import compute_graph_embedding, compute_graph_similarity


class ModelTrainer:
    """
    模型训练器，用于训练GNN模型
    """
    def __init__(self, model, device=None, task_type='node', 
                 loss_fn=None, optimizer=None, lr=0.001, weight_decay=5e-4):
        """
        初始化模型训练器
        
        Args:
            model (torch.nn.Module): GNN模型
            device (torch.device, optional): 计算设备（CPU或GPU）
            task_type (str): 任务类型，'node'表示节点级任务，'graph'表示图级任务
            loss_fn (torch.nn.Module, optional): 损失函数
            optimizer (torch.optim.Optimizer, optional): 优化器
            lr (float): 学习率，仅当optimizer为None时使用
            weight_decay (float): 权重衰减，仅当optimizer为None时使用
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 设置模型
        self.model = model.to(self.device)
        
        # 设置任务类型
        assert task_type in ['node', 'graph'], "任务类型必须是'node'或'graph'"
        self.task_type = task_type
        
        # 设置损失函数
        if loss_fn is None:
            if task_type == 'node':
                # 对于节点分类任务，默认使用交叉熵损失
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                # 对于图分类/回归任务，默认使用MSE损失
                self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = loss_fn
        
        # 设置优化器
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_metric = float('-inf')  # 对于需要最大化的指标（如准确率）
        self.best_model_state = None
    
    def _get_forward_args(self, batch):
        """根据模型的 forward 签名和 batch 内容决定传递哪些参数"""
        forward_params = inspect.signature(self.model.forward).parameters
        args = {'x': batch.x, 'edge_index': batch.edge_index}
        
        if 'edge_attr' in forward_params and hasattr(batch, 'edge_attr'):
            args['edge_attr'] = batch.edge_attr
        if 'edge_type' in forward_params and hasattr(batch, 'edge_type'):
            args['edge_type'] = batch.edge_type
            
        return args

    def train_epoch(self, train_loader, metric_fn=None):
        """
        训练一个epoch
        
        Args:
            train_loader (DataLoader): 训练数据加载器
            metric_fn (callable, optional): 评估指标函数
            
        Returns:
            tuple: (平均损失, 评估指标)
        """
        self.model.train()
        total_loss = 0
        total_metric = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            # 将数据移动到设备
            batch = batch.to(self.device)
            
            # 获取模型需要的参数
            forward_args = self._get_forward_args(batch)
            
            # 前向传播
            out = self.model(**forward_args) # 使用动态参数调用
            
            # 计算损失
            if self.task_type == 'node':
                if hasattr(batch, 'train_mask'):
                    mask = batch.train_mask
                    loss = self.loss_fn(out[mask], batch.y[mask])
                else:
                    loss = self.loss_fn(out, batch.y)
            else: # graph task
                if hasattr(batch, 'batch'):
                    from torch_geometric.nn import global_mean_pool
                    out = global_mean_pool(out, batch.batch)
                loss = self.loss_fn(out, batch.y)

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 记录损失和指标
            total_loss += loss.item()
            num_batches += 1
            
            # 计算指标（如果提供了指标函数）
            if metric_fn is not None:
                with torch.no_grad():
                    # 注意: metric_fn可能也需要masking，这里暂时简化
                    metric = metric_fn(out, batch.y) 
                    total_metric += metric
        
        # 计算平均损失和指标
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        avg_metric = None
        if metric_fn is not None:
            avg_metric = total_metric / num_batches
            self.train_metrics.append(avg_metric)
        
        return avg_loss, avg_metric
    
    def validate(self, val_loader, metric_fn=None):
        """
        在验证集上评估模型
        
        Args:
            val_loader (DataLoader): 验证数据加载器
            metric_fn (callable, optional): 评估指标函数
            
        Returns:
            tuple: (平均损失, 评估指标)
        """
        self.model.eval()
        total_loss = 0
        total_metric = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # 将数据移动到设备
                batch = batch.to(self.device)
                
                # 获取模型需要的参数
                forward_args = self._get_forward_args(batch)

                # 前向传播
                out = self.model(**forward_args) # 使用动态参数调用

                # 计算损失
                if self.task_type == 'node':
                    if hasattr(batch, 'val_mask'):
                        mask = batch.val_mask
                        loss = self.loss_fn(out[mask], batch.y[mask])
                    else:
                        loss = self.loss_fn(out, batch.y)
                else: # graph task
                    if hasattr(batch, 'batch'):
                        from torch_geometric.nn import global_mean_pool
                        out = global_mean_pool(out, batch.batch)
                    loss = self.loss_fn(out, batch.y)
                
                # 记录损失和指标
                total_loss += loss.item()
                num_batches += 1
                
                # 计算指标（如果提供了指标函数）
                if metric_fn is not None:
                     # 注意: metric_fn可能也需要masking，这里暂时简化
                    metric = metric_fn(out, batch.y)
                    total_metric += metric
        
        # 计算平均损失和指标
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        avg_metric = None
        if metric_fn is not None:
            avg_metric = total_metric / num_batches
            self.val_metrics.append(avg_metric)
            
            # 保存最佳模型
            if avg_metric > self.best_val_metric:
                self.best_val_metric = avg_metric
                self.best_model_state = self.model.state_dict().copy()
        
        return avg_loss, avg_metric
    
    def train(self, train_loader, val_loader=None, num_epochs=100, 
             metric_fn=None, early_stopping=None, save_dir=None):
        """
        训练模型
        
        Args:
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader, optional): 验证数据加载器
            num_epochs (int): 训练轮数
            metric_fn (callable, optional): 评估指标函数
            early_stopping (int, optional): 早停轮数，如果验证指标连续多轮不提升则停止训练
            save_dir (str, optional): 模型保存目录
            
        Returns:
            dict: 训练历史
        """
        # 记录训练开始时间
        start_time = time.time()
        
        # 创建保存目录
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
        # 初始化早停计数器
        es_counter = 0
        
        # 训练循环
        for epoch in range(num_epochs):
            # 训练一个epoch
            train_loss, train_metric = self.train_epoch(train_loader, metric_fn)
            
            # 在验证集上评估
            val_loss, val_metric = None, None
            if val_loader is not None:
                val_loss, val_metric = self.validate(val_loader, metric_fn)
                
                # 早停
                if early_stopping is not None:
                    if len(self.val_metrics) > 1 and val_metric <= self.val_metrics[-2]:
                        es_counter += 1
                    else:
                        es_counter = 0
                    
                    if es_counter >= early_stopping:
                        print(f"早停: 验证指标连续{early_stopping}轮未提升")
                        break
            
            # 打印训练进度
            train_metric_str = f"{train_metric:.4f}" if train_metric is not None else 'N/A'
            val_loss_str = f"{val_loss:.4f}" if val_loss is not None else 'N/A'
            val_metric_str = f"{val_metric:.4f}" if val_metric is not None else 'N/A'

            print(f"Epoch {epoch+1}/{num_epochs}, "
                 f"Train Loss: {train_loss:.4f}, "
                 f"Train Metric: {train_metric_str}, "
                 f"Val Loss: {val_loss_str}, "
                 f"Val Metric: {val_metric_str}")
            
            # 保存当前模型
            if save_dir is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))
        
        # 记录训练结束时间
        training_time = time.time() - start_time
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
            # 保存最佳模型
            if save_dir is not None:
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'best_val_metric': self.best_val_metric,
                }, os.path.join(save_dir, "best_model.pt"))
        
        # 返回训练历史
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_metric': self.best_val_metric,
            'training_time': training_time
        }
    
    def predict(self, loader):
        """
        使用模型进行预测
        
        Args:
            loader (DataLoader): 数据加载器
            
        Returns:
            torch.Tensor: 预测结果
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                # 将数据移动到设备
                batch = batch.to(self.device)
                
                # 获取模型需要的参数
                forward_args = self._get_forward_args(batch)
                
                # 前向传播
                out = self.model(**forward_args) # 使用动态参数调用
                
                # 如果是图级任务且模型输出节点嵌入，则进行池化
                if self.task_type == 'graph' and hasattr(batch, 'batch'):
                    from torch_geometric.nn import global_mean_pool
                    out = global_mean_pool(out, batch.batch)
                
                predictions.append(out.cpu())
        
        return torch.cat(predictions, dim=0)
    
    def get_embeddings(self, loader, level='node'):
        """
        获取节点或图嵌入
        
        Args:
            loader (DataLoader): 数据加载器
            level (str): 嵌入级别，'node'表示节点嵌入，'graph'表示图嵌入
            
        Returns:
            torch.Tensor: 嵌入向量
        """
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch in loader:
                # 将数据移动到设备
                batch = batch.to(self.device)
                
                # 获取模型需要的参数
                forward_args = self._get_forward_args(batch)
                
                # 获取节点嵌入 (假设模型总是先输出节点嵌入)
                node_embeddings = self.model(**forward_args) # 使用动态参数调用
                
                if level == 'node':
                    embeddings.append(node_embeddings.cpu())
                else: # graph level
                    if hasattr(batch, 'batch'):
                        from torch_geometric.nn import global_mean_pool
                        graph_embedding = global_mean_pool(node_embeddings, batch.batch)
                    else:
                        # 如果没有batch属性，可能是一个单独的图，需要自定义聚合
                        graph_embedding = compute_graph_embedding(node_embeddings) # 使用已有的工具函数
                    
                    embeddings.append(graph_embedding.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def compute_similarity_matrix(self, embeddings1, embeddings2, method='cosine'):
        """
        计算相似度矩阵
        
        Args:
            embeddings1 (torch.Tensor): 第一组嵌入
            embeddings2 (torch.Tensor): 第二组嵌入
            method (str): 相似度度量方法
            
        Returns:
            torch.Tensor: 相似度矩阵
        """
        return compute_graph_similarity(embeddings1, embeddings2, method)
    
    def load_model(self, model_path):
        """
        加载模型
        
        Args:
            model_path (str): 模型路径
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'best_val_metric' in checkpoint:
            self.best_val_metric = checkpoint['best_val_metric']
        
        return self 