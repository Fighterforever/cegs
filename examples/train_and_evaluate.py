import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

# 导入自定义模块
import sys
sys.path.append('..')
from cegs_gnn.models.baseline import BaselineSAGE
from cegs_gnn.models.rgcn import RGCN
from cegs_gnn.models.edge_sage import EdgeCondSAGE
from cegs_gnn.data_utils.data_loader import prepare_data_for_models
from cegs_gnn.training.trainer import ModelTrainer
from cegs_gnn.evaluation.evaluator import ModelEvaluator
from cegs_gnn.evaluation.metrics import accuracy, f1 