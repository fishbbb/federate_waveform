# -*- coding: utf-8 -*-
"""
硬件分析模块
用于分析设备性能、资源使用情况，支持异构设备
"""

import os
import platform
import psutil
import torch
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class DeviceProfile:
    """设备性能配置文件"""
    device_id: str
    device_type: str  # 'cpu', 'gpu', 'edge'
    cpu_cores: int
    memory_gb: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    compute_capability: Optional[str] = None
    performance_tier: str = 'medium'  # 'low', 'medium', 'high'
    network_bandwidth_mbps: Optional[float] = None
    latency_ms: Optional[float] = None


class DeviceProfiler:
    """设备性能分析器"""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_device(self, device_id: str = 'default') -> DeviceProfile:
        """
        分析设备性能
        
        Args:
            device_id: 设备标识符
            
        Returns:
            DeviceProfile: 设备性能配置
        """
        # CPU信息
        cpu_cores = os.cpu_count() or 1
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        # GPU信息
        gpu_available = torch.cuda.is_available()
        gpu_name = None
        gpu_memory_gb = None
        compute_capability = None
        
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            # 获取计算能力
            major, minor = torch.cuda.get_device_capability(0)
            compute_capability = f"{major}.{minor}"
        
        # 确定设备类型
        device_type = 'gpu' if gpu_available else 'cpu'
        
        # 确定性能等级
        performance_tier = self._classify_performance(
            cpu_cores, memory_gb, gpu_available, gpu_memory_gb
        )
        
        profile = DeviceProfile(
            device_id=device_id,
            device_type=device_type,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            compute_capability=compute_capability,
            performance_tier=performance_tier
        )
        
        self.profiles[device_id] = profile
        return profile
    
    def _classify_performance(self, cpu_cores, memory_gb, gpu_available, gpu_memory_gb):
        """
        根据硬件配置分类性能等级
        
        Returns:
            str: 'low', 'medium', 'high'
        """
        if gpu_available:
            if gpu_memory_gb and gpu_memory_gb >= 8:
                return 'high'
            elif gpu_memory_gb and gpu_memory_gb >= 4:
                return 'medium'
            else:
                return 'low'
        else:
            if cpu_cores >= 8 and memory_gb >= 16:
                return 'high'
            elif cpu_cores >= 4 and memory_gb >= 8:
                return 'medium'
            else:
                return 'low'
    
    def benchmark_device(self, device_id: str, model_size_mb: float = 10.0) -> Dict[str, Any]:
        """
        对设备进行基准测试
        
        Args:
            device_id: 设备标识符
            model_size_mb: 模型大小（MB）
            
        Returns:
            dict: 基准测试结果
        """
        profile = self.profiles.get(device_id)
        if not profile:
            profile = self.profile_device(device_id)
        
        results = {
            'device_id': device_id,
            'profile': profile,
            'benchmark_results': {}
        }
        
        # CPU基准测试
        if profile.device_type == 'cpu':
            cpu_time = self._benchmark_cpu()
            results['benchmark_results']['cpu_inference_time'] = cpu_time
        
        # GPU基准测试
        if profile.gpu_available:
            gpu_time = self._benchmark_gpu()
            results['benchmark_results']['gpu_inference_time'] = gpu_time
        
        # 内存使用测试
        memory_usage = self._test_memory_usage(model_size_mb)
        results['benchmark_results']['memory_usage'] = memory_usage
        
        return results
    
    def _benchmark_cpu(self) -> float:
        """CPU基准测试"""
        # 简单的CPU计算测试
        start = time.time()
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        z = torch.matmul(x, y)
        end = time.time()
        return end - start
    
    def _benchmark_gpu(self) -> float:
        """GPU基准测试"""
        if not torch.cuda.is_available():
            return None
        
        device = torch.device('cuda')
        start = time.time()
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        end = time.time()
        return end - start
    
    def _test_memory_usage(self, model_size_mb: float) -> Dict[str, float]:
        """测试内存使用"""
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 ** 2)  # MB
        
        # 模拟加载模型
        dummy_model = torch.nn.Linear(1000, 1000)
        memory_after = process.memory_info().rss / (1024 ** 2)  # MB
        
        return {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_increase_mb': memory_after - memory_before
        }
    
    def get_all_profiles(self) -> Dict[str, DeviceProfile]:
        """获取所有设备配置"""
        return self.profiles
    
    def get_device_recommendations(self, device_id: str) -> Dict[str, Any]:
        """
        根据设备配置提供训练建议
        
        Args:
            device_id: 设备标识符
            
        Returns:
            dict: 训练建议（批次大小、模型大小等）
        """
        profile = self.profiles.get(device_id)
        if not profile:
            profile = self.profile_device(device_id)
        
        recommendations = {
            'device_id': device_id,
            'performance_tier': profile.performance_tier,
            'recommended_batch_size': 32,
            'recommended_model_size': 'medium',
            'use_mixed_precision': False
        }
        
        if profile.performance_tier == 'high':
            recommendations['recommended_batch_size'] = 128
            recommendations['recommended_model_size'] = 'large'
            recommendations['use_mixed_precision'] = True
        elif profile.performance_tier == 'medium':
            recommendations['recommended_batch_size'] = 64
            recommendations['recommended_model_size'] = 'medium'
        else:  # low
            recommendations['recommended_batch_size'] = 16
            recommendations['recommended_model_size'] = 'small'
        
        return recommendations
