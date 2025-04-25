import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from ..data_utils.graph_utils import compute_graph_similarity, compute_node_similarity, find_node_mapping
from . import metrics


class ModelEvaluator:
    """
    模型评估器，用于评估GNN模型
    """
    def __init__(self, trainers=None, task_type='node'):
        """
        初始化模型评估器
        
        Args:
            trainers (dict, optional): 训练好的模型训练器字典，键为模型名称，值为ModelTrainer对象
            task_type (str): 任务类型，'node'表示节点级任务，'graph'表示图级任务
        """
        self.trainers = trainers if trainers is not None else {}
        self.task_type = task_type
        self.evaluation_results = {}
        
    def add_model(self, name, trainer):
        """
        添加模型
        
        Args:
            name (str): 模型名称
            trainer (ModelTrainer): 训练好的模型训练器
        """
        self.trainers[name] = trainer
        
    def evaluate_example_recommendation(self, test_loader, target_indices=None):
        """
        评估示例推荐任务 (Querier)
        
        Args:
            test_loader (DataLoader): 测试数据加载器
            target_indices (torch.Tensor, optional): 目标图的索引，如果未提供则从数据中获取
            
        Returns:
            dict: 评估结果
        """
        results = {}
        
        for name, trainer in self.trainers.items():
            print(f"评估模型: {name}")
            
            # 测量推理时间
            start_time = time.time()
            
            # 获取图嵌入
            graph_embeddings = trainer.get_embeddings(test_loader, level='graph')
            
            inference_time = time.time() - start_time
            
            # 计算图相似度矩阵
            similarity_matrix = compute_graph_similarity(graph_embeddings, graph_embeddings, method='cosine')
            
            # 如果未提供目标索引，则假设每个图与自身最相似
            if target_indices is None:
                target_indices = torch.arange(graph_embeddings.size(0))
            
            # 计算评估指标
            top1_accuracy = metrics.graph_similarity_accuracy(similarity_matrix, target_indices)
            top5_accuracy = metrics.top_k_accuracy(similarity_matrix, target_indices, k=5)
            mrr = metrics.mean_reciprocal_rank(similarity_matrix, target_indices)
            
            # 记录结果
            results[name] = {
                'top1_accuracy': top1_accuracy,
                'top5_accuracy': top5_accuracy,
                'mrr': mrr,
                'inference_time': inference_time,
                'parameters': metrics.count_parameters(trainer.model)
            }
            
            print(f"  Top-1准确率: {top1_accuracy:.4f}")
            print(f"  Top-5准确率: {top5_accuracy:.4f}")
            print(f"  MRR: {mrr:.4f}")
            print(f"  推理时间: {inference_time:.4f}秒")
            print(f"  参数量: {metrics.count_parameters(trainer.model)}")
        
        # 保存评估结果
        self.evaluation_results['example_recommendation'] = results
        
        return results
    
    def evaluate_node_mapping(self, source_graph, target_graph, true_mapping=None):
        """
        评估节点映射任务 (Classifier)
        
        Args:
            source_graph (Data): 源图
            target_graph (Data): 目标图
            true_mapping (list): 真实的节点映射，每个元素为 (source_idx, target_idx)
            
        Returns:
            dict: 评估结果
        """
        results = {}
        
        for name, trainer in self.trainers.items():
            print(f"评估模型: {name}")
            
            # 测量推理时间
            start_time = time.time()
            
            # 获取源图和目标图的节点嵌入
            with torch.no_grad():
                # 处理源图
                source_graph = source_graph.to(trainer.device)
                source_embeddings = trainer.model(
                    source_graph.x, 
                    source_graph.edge_index,
                    edge_attr=source_graph.edge_attr if hasattr(source_graph, 'edge_attr') else None,
                    edge_type=source_graph.edge_type if hasattr(source_graph, 'edge_type') else None
                )
                
                # 处理目标图
                target_graph = target_graph.to(trainer.device)
                target_embeddings = trainer.model(
                    target_graph.x, 
                    target_graph.edge_index,
                    edge_attr=target_graph.edge_attr if hasattr(target_graph, 'edge_attr') else None,
                    edge_type=target_graph.edge_type if hasattr(target_graph, 'edge_type') else None
                )
            
            # 计算节点相似度矩阵
            similarity_matrix = compute_node_similarity(source_embeddings, target_embeddings, method='cosine')
            
            # 找到最佳节点映射
            pred_mapping = find_node_mapping(similarity_matrix, method='hungarian')
            
            inference_time = time.time() - start_time
            
            # 计算映射准确率（如果提供了真实映射）
            mapping_accuracy = None
            if true_mapping is not None:
                mapping_accuracy = metrics.node_mapping_accuracy(pred_mapping, true_mapping)
            
            # 记录结果
            results[name] = {
                'mapping_accuracy': mapping_accuracy,
                'inference_time': inference_time,
                'parameters': metrics.count_parameters(trainer.model),
                'pred_mapping': pred_mapping
            }
            
            if mapping_accuracy is not None:
                print(f"  映射准确率: {mapping_accuracy:.4f}")
            print(f"  推理时间: {inference_time:.4f}秒")
            print(f"  参数量: {metrics.count_parameters(trainer.model)}")
        
        # 保存评估结果
        self.evaluation_results['node_mapping'] = results
        
        return results
    
    def compare_model_performance(self, task='example_recommendation'):
        """
        比较不同模型的性能
        
        Args:
            task (str): 任务类型，'example_recommendation' 或 'node_mapping'
            
        Returns:
            DataFrame: 性能比较表格
        """
        if task not in self.evaluation_results:
            raise ValueError(f"尚未评估任务: {task}")
        
        results = self.evaluation_results[task]
        
        # 构建比较表格
        data = []
        for model_name, model_results in results.items():
            row = {'Model': model_name}
            row.update(model_results)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 设置显示格式
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        return df
    
    def visualize_node_embeddings(self, graph, method='tsne', save_path=None):
        """
        Visualize node embeddings
        
        Args:
            graph (Data): Graph data
            method (str): Dimensionality reduction method, currently only supports 'tsne'
            save_path (str, optional): Path to save the image
        """
        plt.figure(figsize=(15, 5 * len(self.trainers)))
        
        for i, (name, trainer) in enumerate(self.trainers.items()):
            # Get node embeddings
            with torch.no_grad():
                # Reuse the trainer's get_embeddings logic, which handles forward args
                embeddings = trainer.get_embeddings(graph.to(trainer.device), level='node').cpu().numpy()
            
            # Use t-SNE for dimensionality reduction
            if method == 'tsne':
                tsne = TSNE(n_components=2, random_state=42)
                embeddings_2d = tsne.fit_transform(embeddings)
            else:
                raise ValueError(f"Unsupported dimensionality reduction method: {method}")
            
            # Plot scatter plot
            plt.subplot(len(self.trainers), 1, i+1)
            
            # If the graph has node labels, use different colors for different classes
            if hasattr(graph, 'y'):
                labels = graph.y.cpu().numpy()
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    idx = labels == label
                    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=f'Class {label}')
                plt.legend()
            else:
                plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
            
            plt.title(f"{name} - Node Embedding Visualization")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_training_history(self, save_path=None):
        """
        Visualize training history
        
        Args:
            save_path (str, optional): Path to save the image
        """
        # Check if there is training history
        has_history = False
        for trainer in self.trainers.values():
            if hasattr(trainer, 'train_losses') and len(trainer.train_losses) > 0:
                has_history = True
                break
        
        if not has_history:
            print("No training history to visualize")
            return
        
        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot loss curves
        for name, trainer in self.trainers.items():
            if hasattr(trainer, 'train_losses') and len(trainer.train_losses) > 0:
                epochs = range(1, len(trainer.train_losses) + 1)
                axes[0].plot(epochs, trainer.train_losses, 'o-', label=f'{name} Train')
                
                if hasattr(trainer, 'val_losses') and len(trainer.val_losses) > 0:
                    axes[0].plot(epochs, trainer.val_losses, 's-', label=f'{name} Validation')
        
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot metric curves
        for name, trainer in self.trainers.items():
            if hasattr(trainer, 'train_metrics') and len(trainer.train_metrics) > 0:
                epochs = range(1, len(trainer.train_metrics) + 1)
                axes[1].plot(epochs, trainer.train_metrics, 'o-', label=f'{name} Train')
                
                if hasattr(trainer, 'val_metrics') and len(trainer.val_metrics) > 0:
                    axes[1].plot(epochs, trainer.val_metrics, 's-', label=f'{name} Validation')
        
        axes[1].set_title('Training and Validation Metric')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Metric')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_results(self, output_file):
        """
        导出评估结果
        
        Args:
            output_file (str): 输出文件路径
        """
        results = {}
        
        # 整理评估结果
        for task, task_results in self.evaluation_results.items():
            task_data = {}
            
            for model_name, model_results in task_results.items():
                model_data = {}
                
                for metric, value in model_results.items():
                    if isinstance(value, (int, float)):
                        model_data[metric] = value
                
                task_data[model_name] = model_data
            
            results[task] = task_data
        
        # 将结果导出为CSV文件
        for task, task_data in results.items():
            df = pd.DataFrame.from_dict(task_data, orient='index')
            df.to_csv(f"{output_file}_{task}.csv")
            
        print(f"结果已导出到 {output_file}_*.csv") 