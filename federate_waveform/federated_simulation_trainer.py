# -*- coding: utf-8 -*-
"""
自研联邦学习训练模块
实现联邦训练循环，不再依赖 Fed-BioMed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time
import random
import yaml
import json
from typing import Dict, List, Optional, Any, Callable
from collections import OrderedDict

# Add the Waveform-classification program directory to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
waveform_program_dir = os.path.join(base_dir, 'Waveform-classification', 'program')
sys.path.insert(0, waveform_program_dir)

# Import model and utilities from the original project
from models import myecgnet
from utils import Loss_cal

# Import dataset from training plan
from federated_hypotension_training_plan import (
    HypotensionDataset,
    get_device,
    load_device_profiles,
    is_available_this_round,
    apply_compute_profile,
    simulate_network_delay,
    get_model_size_bytes
)

DEVICE = get_device()


def convert_to_serializable(obj):
    """将对象转换为可序列化的格式（用于 JSON 保存）"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


print(f"[Federated Trainer] Using device: {DEVICE}")


class FederatedClient:
    """联邦学习客户端抽象"""
    
    def __init__(self, client_id: str, data_path: str, device_profile: Dict, model_args: Dict):
        """
        初始化客户端
        
        Args:
            client_id: 客户端ID (如 'node_1')
            data_path: 数据文件路径
            device_profile: 设备配置
            model_args: 模型参数
        """
        self.client_id = client_id
        self.data_path = data_path
        self.device_profile = device_profile
        self.model_args = model_args
        
        # 加载数据
        self._load_data()
        
        # 初始化模型
        self.model = self._create_model()
        # 确保模型参数是 float32（MPS 不支持 float64）
        self.model = self.model.float()
        self.model.to(DEVICE)
        
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=model_args.get('learning_rate', 4e-5)
        )
        
        # 应用设备配置
        apply_compute_profile(device_profile)
        
        print(f"[Client {client_id}] Initialized with {len(self.train_dataset)} training samples")
        print(f"[Client {client_id}] Device profile: {device_profile.get('type', 'unknown')}")
    
    def _load_data(self):
        """加载训练和验证数据"""
        use_uci2 = self.model_args.get('use_uci2', True)
        uci2_base_dir = self.model_args.get('uci2_base_dir', 'uci2_dataset')
        
        # 解析路径
        if not os.path.isabs(self.data_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            potential_path = os.path.join(base_dir, self.data_path)
            if os.path.exists(potential_path):
                self.data_path = potential_path
        
        if not os.path.isabs(uci2_base_dir):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            potential_uci2 = os.path.join(base_dir, uci2_base_dir)
            if os.path.exists(potential_uci2):
                uci2_base_dir = potential_uci2
        
        # 创建数据集
        self.train_dataset = HypotensionDataset(
            data_path=self.data_path,
            uci2_base_dir=uci2_base_dir,
            train=True,
            use_uci2=use_uci2
        )
        
        self.val_dataset = HypotensionDataset(
            data_path=self.data_path,
            uci2_base_dir=uci2_base_dir,
            train=False,
            use_uci2=use_uci2
        )
    
    def _create_model(self):
        """创建模型"""
        model_variant = self.device_profile.get('model_variant', 'base')
        model = myecgnet(pretrained=False, num_classes=1)
        return model
    
    def get_data_loader(self, train=True, batch_size=None):
        """获取数据加载器"""
        if batch_size is None:
            default_batch_size = self.model_args.get('batch_size', 128)
            max_batch_size = self.device_profile.get('max_batch_size', default_batch_size)
            batch_size = min(default_batch_size, max_batch_size)
        
        dataset = self.train_dataset if train else self.val_dataset
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=0  # 避免多进程问题
        )
    
    def train_epoch(self, epochs=1, progress_callback: Optional[Callable] = None):
        """
        训练一个或多个epoch
        
        Args:
            epochs: 训练轮数
            progress_callback: 进度回调函数 (batch_idx, loss) -> None
            
        Returns:
            dict: 训练指标 {'loss': float, 'samples': int}
        """
        self.model.train()
        train_loader = self.get_data_loader(train=True)
        criterion = Loss_cal(alpha=0.5, gamma=2.0)
        
        total_loss = 0.0
        total_samples = 0
        compute_delay = self.device_profile.get('simulated_compute_delay_ms', 0) / 1000.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # 确保数据类型为 float32（MPS 不支持 float64）
                if data.dtype != torch.float32:
                    data = data.float()
                if target.dtype != torch.float32:
                    target = target.float()
                
                # 移动到设备
                data = data.to(DEVICE)
                target = target.to(DEVICE).view(-1, 1)
                
                # 前向传播
                self.optimizer.zero_grad()
                output, _ = self.model(data)
                loss = criterion(output, target)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 统计
                batch_size = data.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size
                
                # 模拟计算延迟
                if compute_delay > 0:
                    time.sleep(compute_delay)
                
                # 进度回调
                if progress_callback:
                    progress_callback(batch_idx, loss.item())
            
            total_loss += epoch_loss
            total_samples += epoch_samples
        
        avg_loss = total_loss / max(total_samples, 1)
        return {
            'loss': avg_loss,
            'samples': total_samples
        }
    
    def evaluate(self):
        """
        评估模型
        
        Returns:
            dict: 评估指标 {'loss': float, 'accuracy': float, 'f1': float, 'samples': int}
        """
        self.model.eval()
        val_loader = self.get_data_loader(train=False)
        criterion = Loss_cal(alpha=0.5, gamma=2.0)
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                # 确保数据类型为 float32（MPS 不支持 float64）
                if data.dtype != torch.float32:
                    data = data.float()
                if target.dtype != torch.float32:
                    target = target.float()
                
                # 移动到设备
                data = data.to(DEVICE)
                target = target.to(DEVICE).view(-1, 1)
                
                output, _ = self.model(data)
                loss = criterion(output, target)
                
                # 统计
                batch_size = data.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # 计算准确率
                preds = (torch.sigmoid(output) > 0.5).int()
                correct += (preds == target.int()).sum().item()
                
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        
        avg_loss = total_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)
        
        # 计算 F1 score（手动实现，避免依赖 sklearn）
        # F1 = 2 * (precision * recall) / (precision + recall)
        tp = sum(1 for p, t in zip(all_preds, all_targets) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(all_preds, all_targets) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(all_preds, all_targets) if p == 0 and t == 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'samples': total_samples
        }
    
    def get_model_state(self):
        """获取模型状态字典"""
        # 确保所有参数都是 float32（MPS 不支持 float64）
        state_dict = self.model.state_dict()
        state_dict_fp32 = {}
        for key, value in state_dict.items():
            if value.dtype != torch.float32:
                state_dict_fp32[key] = value.float()
            else:
                state_dict_fp32[key] = value
        return state_dict_fp32
    
    def set_model_state(self, state_dict):
        """设置模型状态字典"""
        # 确保所有参数都是 float32（MPS 不支持 float64）
        state_dict_fp32 = {}
        for key, value in state_dict.items():
            if value.dtype != torch.float32:
                state_dict_fp32[key] = value.float()
            else:
                state_dict_fp32[key] = value
        self.model.load_state_dict(state_dict_fp32)


def federated_average(client_models: List[Dict], client_weights: List[float]) -> OrderedDict:
    """
    实现 FedAvg 聚合
    
    Args:
        client_models: 客户端模型状态字典列表
        client_weights: 客户端权重列表（通常为数据量）
        
    Returns:
        聚合后的模型状态字典
    """
    # 归一化权重
    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]
    
    # 初始化聚合模型
    aggregated_state = OrderedDict()
    
    # 获取所有键
    all_keys = set()
    for state_dict in client_models:
        all_keys.update(state_dict.keys())
    
    # 加权平均
    for key in all_keys:
        # 获取第一个模型中的对应张量作为模板
        first_tensor = None
        for state_dict in client_models:
            if key in state_dict:
                first_tensor = state_dict[key]
                break
        
        if first_tensor is None:
            continue
        
        # 确保模板张量是 float32（MPS 不支持 float64）
        if first_tensor.dtype != torch.float32:
            first_tensor = first_tensor.float()
        
        # 初始化聚合张量（在 CPU 上，避免 MPS 类型问题）
        aggregated_tensor = torch.zeros_like(first_tensor, device='cpu', dtype=torch.float32)
        
        # 加权求和（在 CPU 上计算）
        for state_dict, weight in zip(client_models, normalized_weights):
            if key in state_dict:
                tensor = state_dict[key]
                # 确保张量是 float32
                if tensor.dtype != torch.float32:
                    tensor = tensor.float()
                # 移动到 CPU 进行聚合
                tensor_cpu = tensor.cpu() if tensor.device.type != 'cpu' else tensor
                aggregated_tensor += tensor_cpu * weight
        
        aggregated_state[key] = aggregated_tensor
    
    return aggregated_state


def run_federated_training(
    config: Dict,
    monitor_instance=None,
    socketio_instance=None,
    progress_callback: Optional[Callable] = None,
    active_nodes: Optional[List[str]] = None,
    save_results: bool = True,
    results_dir: str = 'results',
    enable_compression: bool = False,
    compression_config: Optional[Dict] = None
):
    """
    运行联邦训练
    
    Args:
        config: 训练配置
            - round_limit: 联邦轮数
            - training_args: 训练参数 {'epochs': int, 'batch_size': int, 'learning_rate': float}
            - model_args: 模型参数 {'use_uci2': bool, 'uci2_base_dir': str}
        monitor_instance: 监控实例（可选）
        socketio_instance: SocketIO 实例（可选，用于实时推送）
        progress_callback: 进度回调函数（可选）
        active_nodes: 激活的节点列表（可选）
        save_results: 是否保存结果到 results 文件夹（默认 True）
        results_dir: 结果保存目录（默认 'results'）
        enable_compression: 是否启用模型压缩（默认 False）
        compression_config: 压缩配置字典
            - quantization: {'enabled': bool, 'type': 'dynamic'|'static'}
            - pruning: {'enabled': bool, 'ratio': float, 'type': str}
        
    Returns:
        dict: 训练结果，包含：
            - history: 训练历史
            - final_metrics: 最终指标
            - global_model: 最终全局模型
            - best_model: 最佳模型
            - best_round: 最佳模型所在轮次
            - best_metric: 最佳指标值
    """
    # 解析配置
    rounds = config.get('round_limit', 5)
    training_args = config.get('training_args', {})
    model_args = config.get('model_args', {})
    
    epochs_per_round = training_args.get('epochs', 1)
    learning_rate = training_args.get('optimizer_args', {}).get('lr', 4e-5)
    batch_size = training_args.get('loader_args', {}).get('batch_size', 128)
    
    # 更新 model_args
    model_args['batch_size'] = batch_size
    model_args['learning_rate'] = learning_rate
    
    # 加载设备配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    devices_yaml = os.path.join(current_dir, 'devices.yaml')
    device_profiles = load_device_profiles(devices_yaml)
    
    # 创建客户端
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clients = []
    active_clients = []
    
    # 确定要创建的节点列表
    if active_nodes is None:
        # 如果没有指定，尝试从 monitor_instance 获取
        if monitor_instance and hasattr(monitor_instance, 'state'):
            active_nodes = list(monitor_instance.state.get('active_nodes', set()))
        else:
            # 默认创建所有节点
            active_nodes = ['node_1', 'node_2', 'node_3']
    
    # 如果没有激活的节点，使用默认列表
    if not active_nodes:
        active_nodes = ['node_1', 'node_2', 'node_3']
        print(f"[Warning] No active nodes specified, using default: {active_nodes}")
    else:
        # 节点列表保持原顺序（将在训练时按 online_pattern 优先级排序）
        print(f"[Federated Trainer] Creating clients for active nodes: {active_nodes}")
    
    for node_id in active_nodes:
        # 查找设备配置（通过 fb_node_path 匹配）
        device_profile = None
        fb_node_path = f'fbm-node-{node_id.split("_")[-1]}'
        for profile in device_profiles:
            if profile.get('id') == node_id or profile.get('fb_node_path') == fb_node_path:
                device_profile = profile
                break
        
        if device_profile is None:
            print(f"[Warning] No device profile found for {node_id}, using default")
            # 使用默认配置
            device_profile = {
                "id": node_id,
                "fb_node_path": fb_node_path,
                "type": "default",
                "compute_power": "medium",
                "cpu_threads": 2,
                "max_batch_size": 64,
                "simulated_compute_delay_ms": 0,
                "upload_latency_ms": 0,
                "download_latency_ms": 0,
                "bandwidth_kbps": 1000,
                "online_pattern": "always_on",
                "model_variant": "base"
            }
        
        # 数据路径
        data_path = os.path.join(base_dir, 'federated_data', node_id, 'train.pth')
        if not os.path.exists(data_path):
            print(f"[Warning] Data not found for {node_id} at {data_path}, skipping this client")
            print(f"  Expected path: {data_path}")
            continue
        else:
            print(f"[Federated Trainer] Found data for {node_id} at {data_path}")
        
        # 创建客户端
        try:
            client = FederatedClient(
                client_id=node_id,
                data_path=data_path,
                device_profile=device_profile,
                model_args=model_args
            )
            clients.append(client)
            active_clients.append(node_id)
            online_pattern = device_profile.get('online_pattern', 'unknown')
            print(f"[Federated Trainer] Client {node_id} initialized successfully (online_pattern: {online_pattern})")
        except Exception as e:
            print(f"[Error] Failed to initialize client {node_id}: {e}")
            continue
    
    if len(clients) == 0:
        raise ValueError("No clients available for training")
    
    print(f"[Federated Trainer] Starting federated training with {len(clients)} clients")
    print(f"[Federated Trainer] Total rounds: {rounds}, Epochs per round: {epochs_per_round}")
    
    # 初始化全局模型（使用第一个客户端的模型）
    global_model_state = clients[0].get_model_state()
    # 确保全局模型状态在 CPU 上且为 float32
    for key in global_model_state:
        if global_model_state[key].device.type != 'cpu':
            global_model_state[key] = global_model_state[key].cpu()
        if global_model_state[key].dtype != torch.float32:
            global_model_state[key] = global_model_state[key].float()
    
    for client in clients:
        client.set_model_state(global_model_state)
        # 确保模型在正确的设备上
        client.model.to(DEVICE)
    
    # 训练历史
    training_history = []
    
    # 设置结果保存目录
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'training_data'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'compressed_models'), exist_ok=True)
        print(f"[Federated Trainer] Results will be saved to: {results_dir}/")
    
    # 最佳模型跟踪
    best_metric_value = -1.0  # 使用 F1 score 作为主要指标
    best_round = -1
    best_model_state = None
    
    # 导入模型压缩模块（如果需要）
    if enable_compression:
        from model_quantization import ModelQuantizer
        from model_pruning import ModelPruner
        from model_distillation import KnowledgeDistillation
        quantizer = ModelQuantizer()
        pruner = ModelPruner()
        compression_config = compression_config or {}
    
    # 联邦训练循环
    for round_num in range(rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_num + 1}/{rounds}")
        print(f"{'='*60}")
        
        # 在训练开始前通知 monitor
        if monitor_instance:
            monitor_instance.start_round(round_num)
        
        # 推送状态更新（轮次开始）
        if socketio_instance and monitor_instance:
            socketio_instance.emit('state_update', monitor_instance.get_state())
            socketio_instance.emit('update', {
                'event_type': 'round_started',
                'data': {'round': round_num},
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        round_start_time = time.time()
        
        # 选择参与训练的客户端（根据在线模式）
        participating_clients = []
        
        # 定义 online_pattern 的优先级（可靠性从高到低）
        pattern_priority = {
            'always_on': 0,        # 最高优先级
            'mostly_online': 1,
            'intermittent': 2,
            'sporadic': 3,         # 最低优先级
        }
        
        # 定义 compute_power 的优先级（从高到低）
        compute_priority = {
            'high': 0,
            'medium': 1,
            'low': 2,
        }
        
        # 按 online_pattern 优先级排序，相同优先级时按 compute_power 排序
        def get_client_priority(client):
            pattern = client.device_profile.get('online_pattern', 'unknown')
            compute = client.device_profile.get('compute_power', 'unknown')
            pattern_prio = pattern_priority.get(pattern, 999)
            compute_prio = compute_priority.get(compute, 999)
            return (pattern_prio, compute_prio)
        
        sorted_clients = sorted(clients, key=get_client_priority)
        for client in sorted_clients:
            is_available = is_available_this_round(client.device_profile, round_num)
            online_pattern = client.device_profile.get('online_pattern', 'unknown')
            compute_power = client.device_profile.get('compute_power', 'unknown')
            print(f"[Round {round_num + 1}] Client {client.client_id} (pattern: {online_pattern}, power: {compute_power}): {'✓ available' if is_available else '✗ unavailable'}")
            if is_available:
                participating_clients.append(client)
        
        if len(participating_clients) == 0:
            print(f"[Warning] No clients available in round {round_num + 1}, skipping")
            continue
        
        # 参与训练的客户端也按相同优先级排序
        participating_clients = sorted(participating_clients, key=get_client_priority)
        print(f"[Round {round_num + 1}] {len(participating_clients)} clients participating (in priority order): {[c.client_id for c in participating_clients]}")
        
        # 客户端本地训练
        client_models = []
        client_weights = []
        client_metrics = {}
        
        for client in participating_clients:
            print(f"[Round {round_num + 1}] Training client {client.client_id}...")
            
            # 更新节点状态为训练中
            if monitor_instance:
                monitor_instance.update_node_status(
                    client.client_id,  # node_id (位置参数)
                    'training',  # status (位置参数)
                    metrics={}  # kwargs
                )
            
            # 推送状态更新（节点开始训练）
            if socketio_instance and monitor_instance:
                socketio_instance.emit('state_update', monitor_instance.get_state())
            
            client_start_time = time.time()
            
            # 本地训练
            train_metrics = client.train_epoch(epochs=epochs_per_round)
            
            # 评估
            val_metrics = client.evaluate()
            
            # 获取模型和权重
            client_models.append(client.get_model_state())
            client_weights.append(train_metrics['samples'])
            
            # 记录指标
            client_metrics[client.client_id] = {
                'loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'accuracy': val_metrics['accuracy'],
                'f1': val_metrics['f1'],
                'samples': train_metrics['samples']
            }
            
            client_time = time.time() - client_start_time
            print(f"[Round {round_num + 1}] Client {client.client_id} completed in {client_time:.2f}s")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            # 模拟网络延迟（上传模型）
            model_size = get_model_size_bytes(client.get_model_state())
            simulate_network_delay(client.device_profile, model_size)
        
        # FedAvg 聚合
        print(f"[Round {round_num + 1}] Aggregating models...")
        global_model_state = federated_average(client_models, client_weights)
        
        # 确保全局模型状态在 CPU 上且为 float32（federated_average 已经处理，但再次确认）
        for key in global_model_state:
            if global_model_state[key].device.type != 'cpu':
                global_model_state[key] = global_model_state[key].cpu()
            if global_model_state[key].dtype != torch.float32:
                global_model_state[key] = global_model_state[key].float()
        
        # 分发全局模型到所有客户端
        for client in clients:
            client.set_model_state(global_model_state)
            # 确保模型在正确的设备上
            client.model.to(DEVICE)
        
        # 全局评估（在所有客户端上）
        global_metrics = {
            'loss': np.mean([m['loss'] for m in client_metrics.values()]),
            'val_loss': np.mean([m['val_loss'] for m in client_metrics.values()]),
            'accuracy': np.mean([m['accuracy'] for m in client_metrics.values()]),
            'f1': np.mean([m['f1'] for m in client_metrics.values()])
        }
        
        round_time = time.time() - round_start_time
        print(f"[Round {round_num + 1}] Completed in {round_time:.2f}s")
        print(f"  Global Loss: {global_metrics['loss']:.4f}, Global Val Loss: {global_metrics['val_loss']:.4f}")
        print(f"  Global Accuracy: {global_metrics['accuracy']:.4f}, Global F1: {global_metrics['f1']:.4f}")
        
        # 更新监控（在训练完成后）
        if monitor_instance:
            monitor_instance.update_round_metrics(round_num, client_metrics, global_metrics)
            
            # 更新节点状态为完成
            for client_id, metrics in client_metrics.items():
                monitor_instance.update_node_status(
                    client_id,  # node_id (位置参数)
                    'completed',  # status (位置参数)
                    metrics={  # kwargs
                        'loss': metrics['loss'],
                        'f1': metrics['f1'],
                        'accuracy': metrics['accuracy']
                    }
                )
        
        # 推送状态更新（轮次完成）
        if socketio_instance and monitor_instance:
            socketio_instance.emit('state_update', monitor_instance.get_state())
            socketio_instance.emit('update', {
                'event_type': 'round_completed',
                'data': {
                    'round': round_num,
                    'metrics': global_metrics,
                    'client_metrics': client_metrics
                },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # 记录历史
        training_history.append({
            'round': round_num,
            'global_metrics': global_metrics,
            'client_metrics': client_metrics,
            'time': round_time
        })
        
        # 实时保存训练数据到 results 文件夹
        if save_results:
            # 保存当前轮次的训练数据
            round_data_path = os.path.join(results_dir, 'training_data', f'round_{round_num}.json')
            try:
                round_data = {
                    'round': round_num,
                    'global_metrics': convert_to_serializable(global_metrics),
                    'client_metrics': convert_to_serializable(client_metrics),
                    'time': round_time,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                with open(round_data_path, 'w') as f:
                    json.dump(round_data, f, indent=2)
                print(f"[Round {round_num + 1}] Training data saved to: {round_data_path}")
            except Exception as e:
                print(f"[Warning] Failed to save round data: {e}")
            
            # 保存完整的训练历史（每轮更新）
            history_path = os.path.join(results_dir, 'training_data', 'training_history.json')
            try:
                serializable_history = convert_to_serializable(training_history)
                with open(history_path, 'w') as f:
                    json.dump(serializable_history, f, indent=2)
            except Exception as e:
                print(f"[Warning] Failed to save training history: {e}")
        
        # 保存当前模型
        current_metric_value = global_metrics.get('f1', 0.0)  # 使用 F1 score 作为主要指标
        
        if save_results:
            # 保存当前轮次的模型
            current_model_path = os.path.join(results_dir, 'models', f'current_round_{round_num}.pth')
            try:
                torch.save(global_model_state, current_model_path)
                print(f"[Round {round_num + 1}] Current model saved to: {current_model_path}")
            except Exception as e:
                print(f"[Warning] Failed to save current model: {e}")
        
        # 检查是否是最佳模型（基于 F1 score）
        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_round = round_num
            best_model_state = {k: v.clone() for k, v in global_model_state.items()}  # 深拷贝
            
            if save_results:
                # 保存最佳模型
                best_model_path = os.path.join(results_dir, 'models', 'best_model.pth')
                try:
                    torch.save(best_model_state, best_model_path)
                    print(f"[Round {round_num + 1}] ⭐ New best model! (F1: {best_metric_value:.4f})")
                    print(f"[Round {round_num + 1}] Best model saved to: {best_model_path}")
                    
                    # 保存最佳模型信息
                    best_model_info = {
                        'round': best_round,
                        'f1_score': float(best_metric_value),
                        'accuracy': float(global_metrics.get('accuracy', 0.0)),
                        'loss': float(global_metrics.get('loss', 0.0)),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    best_info_path = os.path.join(results_dir, 'models', 'best_model_info.json')
                    with open(best_info_path, 'w') as f:
                        json.dump(best_model_info, f, indent=2)
                except Exception as e:
                    print(f"[Warning] Failed to save best model: {e}")
        
        # 应用模型压缩（如果启用）
        if enable_compression and save_results:
            try:
                # 创建模型实例用于压缩
                from models import myecgnet
                model = myecgnet(pretrained=False, num_classes=1)
                model.load_state_dict(global_model_state)
                model.eval()
                
                compressed_dir = os.path.join(results_dir, 'compressed_models', f'round_{round_num}')
                os.makedirs(compressed_dir, exist_ok=True)
                
                # 量化
                if compression_config.get('quantization', {}).get('enabled', False):
                    quant_type = compression_config['quantization'].get('type', 'dynamic')
                    quantized_model = quantizer.quantize_model(model, quantization_type=quant_type)
                    quantized_path = os.path.join(compressed_dir, 'quantized_model.pth')
                    torch.save(quantized_model.state_dict(), quantized_path)
                    print(f"[Round {round_num + 1}] Quantized model saved to: {quantized_path}")
                
                # 剪枝
                if compression_config.get('pruning', {}).get('enabled', False):
                    pruning_ratio = compression_config['pruning'].get('ratio', 0.3)
                    pruning_type = compression_config['pruning'].get('type', 'l1_unstructured')
                    pruned_model = pruner.prune_model(model, pruning_ratio=pruning_ratio, pruning_type=pruning_type)
                    pruned_path = os.path.join(compressed_dir, 'pruned_model.pth')
                    torch.save(pruned_model.state_dict(), pruned_path)
                    print(f"[Round {round_num + 1}] Pruned model saved to: {pruned_path}")
                
            except Exception as e:
                print(f"[Warning] Model compression failed: {e}")
        
        # 进度回调
        if progress_callback:
            progress_callback(round_num, global_metrics, client_metrics)
    
    print(f"\n{'='*60}")
    print("Federated Training Completed!")
    print(f"{'='*60}")
    
    # 最终评估
    final_metrics = {}
    for client in clients:
        metrics = client.evaluate()
        final_metrics[client.client_id] = metrics
        print(f"Final metrics for {client.client_id}:")
        print(f"  Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    
    # 保存最终结果
    if save_results:
        try:
            # 保存最终指标
            final_metrics_path = os.path.join(results_dir, 'training_data', 'final_metrics.json')
            with open(final_metrics_path, 'w') as f:
                json.dump(convert_to_serializable(final_metrics), f, indent=2)
            print(f"Final metrics saved to: {final_metrics_path}")
            
            # 保存最终模型
            final_model_path = os.path.join(results_dir, 'models', 'final_model.pth')
            torch.save(global_model_state, final_model_path)
            print(f"Final model saved to: {final_model_path}")
            
            # 保存训练总结
            summary = {
                'total_rounds': rounds,
                'best_round': best_round,
                'best_f1_score': float(best_metric_value),
                'final_metrics': convert_to_serializable(final_metrics),
                'training_time': sum([h['time'] for h in training_history]),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            summary_path = os.path.join(results_dir, 'training_data', 'training_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Training summary saved to: {summary_path}")
            
            print(f"\n{'='*60}")
            print(f"Best model achieved at Round {best_round + 1} with F1 Score: {best_metric_value:.4f}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"[Warning] Failed to save final results: {e}")
    
    return {
        'history': training_history,
        'final_metrics': final_metrics,
        'global_model': global_model_state,
        'best_model': best_model_state,
        'best_round': best_round,
        'best_metric': best_metric_value
    }


