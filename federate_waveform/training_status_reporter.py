# -*- coding: utf-8 -*-
"""
训练状态报告器
用于训练脚本向监控系统报告状态
"""

import os
import json
import requests
from typing import Dict, Any, Optional


class TrainingStatusReporter:
    """训练状态报告器"""
    
    def __init__(self, monitor_url: str = 'http://localhost:5002'):
        """
        初始化报告器
        
        Args:
            monitor_url: 监控服务器URL
        """
        self.monitor_url = monitor_url
        self.enabled = True
    
    def report_experiment_started(self, config: Dict[str, Any]):
        """报告实验开始"""
        if not self.enabled:
            return
        
        try:
            # 通过API更新状态
            response = requests.post(
                f'{self.monitor_url}/api/training/status',
                json={'action': 'start', 'config': config},
                timeout=2
            )
        except Exception:
            # 如果监控服务器不可用，静默失败
            pass
    
    def report_round_started(self, round_num: int):
        """报告轮次开始"""
        if not self.enabled:
            return
        
        try:
            response = requests.post(
                f'{self.monitor_url}/api/training/status',
                json={'action': 'round_started', 'round': round_num},
                timeout=2
            )
        except Exception:
            pass
    
    def report_round_metrics(self, round_num: int, metrics: Dict[str, float]):
        """报告轮次指标"""
        if not self.enabled:
            return
        
        try:
            response = requests.post(
                f'{self.monitor_url}/api/training/status',
                json={'action': 'round_metrics', 'round': round_num, 'metrics': metrics},
                timeout=2
            )
        except Exception:
            pass
    
    def report_node_status(self, node_id: str, status: str, metrics: Optional[Dict] = None):
        """报告节点状态"""
        if not self.enabled:
            return
        
        try:
            response = requests.post(
                f'{self.monitor_url}/api/training/status',
                json={'action': 'node_status', 'node_id': node_id, 'status': status, 'metrics': metrics},
                timeout=2
            )
        except Exception:
            pass
    
    def report_experiment_ended(self):
        """报告实验结束"""
        if not self.enabled:
            return
        
        try:
            response = requests.post(
                f'{self.monitor_url}/api/training/status',
                json={'action': 'end'},
                timeout=2
            )
        except Exception:
            pass


# 全局报告器实例
_reporter = None

def get_reporter() -> TrainingStatusReporter:
    """获取全局报告器实例"""
    global _reporter
    if _reporter is None:
        monitor_port = os.environ.get('FL_MONITOR_PORT', '5002')
        monitor_url = f'http://localhost:{monitor_port}'
        _reporter = TrainingStatusReporter(monitor_url)
    return _reporter
