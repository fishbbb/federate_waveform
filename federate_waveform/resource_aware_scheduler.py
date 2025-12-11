# -*- coding: utf-8 -*-
"""
资源感知调度器
根据设备能力和资源状态动态调整训练策略
"""

import time
from typing import Dict, List, Optional, Any
from collections import defaultdict
from hardware_profiling import DeviceProfiler, DeviceProfile


class ResourceAwareScheduler:
    """资源感知调度器"""
    
    def __init__(self, device_profiler: DeviceProfiler):
        """
        初始化调度器
        
        Args:
            device_profiler: 设备分析器实例
        """
        self.device_profiler = device_profiler
        self.device_status = {}  # {device_id: {status, last_update, resources}}
        self.scheduling_history = []
    
    def register_device(self, device_id: str, profile: DeviceProfile):
        """
        注册设备
        
        Args:
            device_id: 设备标识符
            profile: 设备配置
        """
        self.device_status[device_id] = {
            'profile': profile,
            'status': 'idle',
            'last_update': time.time(),
            'resources': {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'gpu_usage': 0.0,
                'network_bandwidth': None
            },
            'current_task': None
        }
    
    def update_device_status(
        self, 
        device_id: str, 
        status: str = 'idle',
        resources: Optional[Dict[str, float]] = None
    ):
        """
        更新设备状态
        
        Args:
            device_id: 设备标识符
            status: 设备状态 ('idle', 'training', 'busy', 'error')
            resources: 资源使用情况
        """
        if device_id not in self.device_status:
            # 如果设备未注册，先分析设备
            profile = self.device_profiler.profile_device(device_id)
            self.register_device(device_id, profile)
        
        self.device_status[device_id]['status'] = status
        self.device_status[device_id]['last_update'] = time.time()
        
        if resources:
            self.device_status[device_id]['resources'].update(resources)
    
    def select_devices_for_training(
        self, 
        num_devices: int,
        priority: str = 'performance'
    ) -> List[str]:
        """
        选择用于训练的设备
        
        Args:
            num_devices: 需要的设备数量
            priority: 优先级策略 ('performance', 'availability', 'balanced')
            
        Returns:
            List[str]: 选中的设备ID列表
        """
        available_devices = [
            device_id for device_id, status in self.device_status.items()
            if status['status'] == 'idle'
        ]
        
        if len(available_devices) < num_devices:
            print(f"警告: 只有 {len(available_devices)} 个可用设备，需要 {num_devices} 个")
            return available_devices
        
        if priority == 'performance':
            # 按性能排序
            sorted_devices = sorted(
                available_devices,
                key=lambda d: self._get_device_score(d),
                reverse=True
            )
        elif priority == 'availability':
            # 按可用性排序（最近更新的优先）
            sorted_devices = sorted(
                available_devices,
                key=lambda d: self.device_status[d]['last_update'],
                reverse=True
            )
        else:  # balanced
            # 平衡策略：考虑性能和可用性
            sorted_devices = sorted(
                available_devices,
                key=lambda d: self._get_balanced_score(d),
                reverse=True
            )
        
        return sorted_devices[:num_devices]
    
    def _get_device_score(self, device_id: str) -> float:
        """计算设备性能分数"""
        profile = self.device_status[device_id]['profile']
        score = 0.0
        
        # CPU分数
        score += profile.cpu_cores * 0.1
        
        # 内存分数
        score += profile.memory_gb * 0.05
        
        # GPU分数
        if profile.gpu_available:
            score += 10.0
            if profile.gpu_memory_gb:
                score += profile.gpu_memory_gb * 0.5
        
        # 性能等级
        tier_scores = {'low': 1.0, 'medium': 2.0, 'high': 3.0}
        score += tier_scores.get(profile.performance_tier, 1.0)
        
        return score
    
    def _get_balanced_score(self, device_id: str) -> float:
        """计算平衡分数（性能和可用性）"""
        performance_score = self._get_device_score(device_id)
        
        # 可用性分数（基于最后更新时间）
        time_since_update = time.time() - self.device_status[device_id]['last_update']
        availability_score = 1.0 / (1.0 + time_since_update / 3600)  # 每小时衰减
        
        return performance_score * 0.7 + availability_score * 0.3
    
    def get_training_config(self, device_id: str) -> Dict[str, Any]:
        """
        根据设备能力获取训练配置
        
        Args:
            device_id: 设备标识符
            
        Returns:
            dict: 训练配置（批次大小、学习率等）
        """
        if device_id not in self.device_status:
            profile = self.device_profiler.profile_device(device_id)
            self.register_device(device_id, profile)
        
        profile = self.device_status[device_id]['profile']
        recommendations = self.device_profiler.get_device_recommendations(device_id)
        
        config = {
            'batch_size': recommendations['recommended_batch_size'],
            'learning_rate': 4e-5,  # 基础学习率
            'num_workers': min(4, profile.cpu_cores // 2),
            'pin_memory': profile.gpu_available,
            'use_mixed_precision': recommendations['use_mixed_precision']
        }
        
        # 根据性能等级调整学习率
        if profile.performance_tier == 'low':
            config['learning_rate'] *= 0.8  # 降低学习率以稳定训练
        elif profile.performance_tier == 'high':
            config['learning_rate'] *= 1.2  # 提高学习率以加快收敛
        
        return config
    
    def schedule_training_task(
        self, 
        task_id: str,
        required_devices: int,
        priority: str = 'performance'
    ) -> Dict[str, Any]:
        """
        调度训练任务
        
        Args:
            task_id: 任务ID
            required_devices: 需要的设备数量
            priority: 优先级策略
            
        Returns:
            dict: 调度结果
        """
        selected_devices = self.select_devices_for_training(required_devices, priority)
        
        if not selected_devices:
            return {
                'success': False,
                'message': '没有可用的设备'
            }
        
        # 为每个设备生成配置
        device_configs = {}
        for device_id in selected_devices:
            device_configs[device_id] = self.get_training_config(device_id)
            self.device_status[device_id]['status'] = 'training'
            self.device_status[device_id]['current_task'] = task_id
        
        scheduling_record = {
            'task_id': task_id,
            'timestamp': time.time(),
            'selected_devices': selected_devices,
            'device_configs': device_configs
        }
        
        self.scheduling_history.append(scheduling_record)
        
        return {
            'success': True,
            'task_id': task_id,
            'devices': selected_devices,
            'configs': device_configs
        }
    
    def release_device(self, device_id: str):
        """释放设备"""
        if device_id in self.device_status:
            self.device_status[device_id]['status'] = 'idle'
            self.device_status[device_id]['current_task'] = None
    
    def get_scheduling_statistics(self) -> Dict[str, Any]:
        """获取调度统计信息"""
        total_devices = len(self.device_status)
        idle_devices = sum(1 for s in self.device_status.values() if s['status'] == 'idle')
        training_devices = sum(1 for s in self.device_status.values() if s['status'] == 'training')
        
        return {
            'total_devices': total_devices,
            'idle_devices': idle_devices,
            'training_devices': training_devices,
            'total_tasks': len(self.scheduling_history),
            'device_utilization': training_devices / total_devices if total_devices > 0 else 0.0
        }
