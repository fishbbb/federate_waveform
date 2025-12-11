# -*- coding: utf-8 -*-
"""
自适应系统模块
根据网络条件和设备可用性动态调整训练策略
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque
from hardware_profiling import DeviceProfiler
from resource_aware_scheduler import ResourceAwareScheduler


class AdaptiveFederatedLearning:
    """自适应联邦学习系统"""
    
    def __init__(
        self,
        device_profiler: DeviceProfiler,
        scheduler: ResourceAwareScheduler
    ):
        """
        初始化自适应系统
        
        Args:
            device_profiler: 设备分析器
            scheduler: 资源感知调度器
        """
        self.device_profiler = device_profiler
        self.scheduler = scheduler
        
        # 网络条件监控
        self.network_conditions = {
            'latency_history': deque(maxlen=100),
            'bandwidth_history': deque(maxlen=100),
            'packet_loss_history': deque(maxlen=100)
        }
        
        # 自适应参数
        self.adaptive_params = {
            'round_interval': 1,  # 训练轮次间隔
            'batch_size': 128,
            'learning_rate': 4e-5,
            'node_selection_threshold': 0.5  # 节点选择阈值
        }
        
        # 历史记录
        self.adaptation_history = []
    
    def monitor_network_conditions(
        self,
        latency_ms: float,
        bandwidth_mbps: Optional[float] = None,
        packet_loss: Optional[float] = None
    ):
        """
        监控网络条件
        
        Args:
            latency_ms: 延迟（毫秒）
            bandwidth_mbps: 带宽（Mbps）
            packet_loss: 丢包率
        """
        self.network_conditions['latency_history'].append(latency_ms)
        if bandwidth_mbps is not None:
            self.network_conditions['bandwidth_history'].append(bandwidth_mbps)
        if packet_loss is not None:
            self.network_conditions['packet_loss_history'].append(packet_loss)
    
    def get_network_status(self) -> Dict[str, Any]:
        """
        获取网络状态
        
        Returns:
            网络状态信息
        """
        latency_history = list(self.network_conditions['latency_history'])
        bandwidth_history = list(self.network_conditions['bandwidth_history'])
        packet_loss_history = list(self.network_conditions['packet_loss_history'])
        
        avg_latency = sum(latency_history) / len(latency_history) if latency_history else 0
        avg_bandwidth = sum(bandwidth_history) / len(bandwidth_history) if bandwidth_history else None
        avg_packet_loss = sum(packet_loss_history) / len(packet_loss_history) if packet_loss_history else None
        
        # 判断网络状态
        if avg_latency > 500 or (avg_packet_loss and avg_packet_loss > 0.1):
            network_status = 'poor'
        elif avg_latency > 200 or (avg_packet_loss and avg_packet_loss > 0.05):
            network_status = 'moderate'
        else:
            network_status = 'good'
        
        return {
            'status': network_status,
            'avg_latency_ms': avg_latency,
            'avg_bandwidth_mbps': avg_bandwidth,
            'avg_packet_loss': avg_packet_loss,
            'samples': len(latency_history)
        }
    
    def adjust_training_strategy(
        self,
        network_status: Optional[Dict[str, Any]] = None,
        device_status: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        根据网络条件和设备状态调整训练策略
        
        Args:
            network_status: 网络状态
            device_status: 设备状态
            
        Returns:
            调整后的训练配置
        """
        if network_status is None:
            network_status = self.get_network_status()
        
        if device_status is None:
            device_status = self.scheduler.get_scheduling_statistics()
        
        # 保存原始参数
        original_params = self.adaptive_params.copy()
        
        # 根据网络条件调整
        if network_status['status'] == 'poor':
            # 网络条件差：减少通信频率，增加本地训练轮次
            self.adaptive_params['round_interval'] = min(
                self.adaptive_params['round_interval'] * 2, 5
            )
            self.adaptive_params['batch_size'] = max(
                int(self.adaptive_params['batch_size'] * 0.8), 32
            )
        elif network_status['status'] == 'good':
            # 网络条件好：可以增加通信频率
            self.adaptive_params['round_interval'] = max(
                self.adaptive_params['round_interval'] * 0.8, 1
            )
            self.adaptive_params['batch_size'] = min(
                int(self.adaptive_params['batch_size'] * 1.1), 256
            )
        
        # 根据设备状态调整
        if device_status['device_utilization'] > 0.8:
            # 设备利用率高：减少批次大小
            self.adaptive_params['batch_size'] = max(
                int(self.adaptive_params['batch_size'] * 0.9), 32
            )
        elif device_status['device_utilization'] < 0.3:
            # 设备利用率低：可以增加批次大小
            self.adaptive_params['batch_size'] = min(
                int(self.adaptive_params['batch_size'] * 1.1), 256
            )
        
        # 记录调整历史
        adaptation_record = {
            'timestamp': time.time(),
            'network_status': network_status,
            'device_status': device_status,
            'original_params': original_params,
            'new_params': self.adaptive_params.copy()
        }
        self.adaptation_history.append(adaptation_record)
        
        return self.adaptive_params.copy()
    
    def select_nodes_adaptively(
        self,
        available_nodes: List[str],
        required_nodes: int
    ) -> List[str]:
        """
        自适应选择节点
        
        Args:
            available_nodes: 可用节点列表
            required_nodes: 需要的节点数量
            
        Returns:
            选中的节点列表
        """
        # 根据网络状态和设备性能选择节点
        network_status = self.get_network_status()
        
        if network_status['status'] == 'poor':
            # 网络条件差：优先选择高性能节点，减少通信次数
            priority = 'performance'
        else:
            # 网络条件好：可以使用平衡策略
            priority = 'balanced'
        
        return self.scheduler.select_devices_for_training(
            required_nodes,
            priority=priority
        )
    
    def adjust_learning_rate(
        self,
        current_round: int,
        total_rounds: int,
        convergence_rate: Optional[float] = None
    ) -> float:
        """
        自适应调整学习率
        
        Args:
            current_round: 当前轮次
            total_rounds: 总轮次
            convergence_rate: 收敛率（可选）
            
        Returns:
            调整后的学习率
        """
        base_lr = self.adaptive_params['learning_rate']
        
        # 学习率衰减（余弦退火）
        if convergence_rate is None:
            # 标准余弦退火
            lr = base_lr * (1 + np.cos(np.pi * current_round / total_rounds)) / 2
        else:
            # 根据收敛率调整
            if convergence_rate < 0.01:  # 收敛缓慢
                lr = base_lr * 1.1  # 稍微增加学习率
            elif convergence_rate > 0.1:  # 收敛过快或不稳定
                lr = base_lr * 0.9  # 降低学习率
            else:
                lr = base_lr
        
        self.adaptive_params['learning_rate'] = lr
        return lr
    
    def get_adaptive_config(self) -> Dict[str, Any]:
        """
        获取当前自适应配置
        
        Returns:
            配置字典
        """
        network_status = self.get_network_status()
        device_status = self.scheduler.get_scheduling_statistics()
        
        return {
            'training_params': self.adaptive_params.copy(),
            'network_status': network_status,
            'device_status': device_status,
            'adaptation_count': len(self.adaptation_history)
        }
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """获取自适应调整历史"""
        return self.adaptation_history.copy()
