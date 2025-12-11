# -*- coding: utf-8 -*-
"""
收敛监控模块
监控训练收敛情况，分析模型参数变化和节点贡献度
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import os


class ConvergenceMonitor:
    """收敛监控器"""
    
    def __init__(self, window_size: int = 10):
        """
        初始化收敛监控器
        
        Args:
            window_size: 滑动窗口大小（用于计算收敛指标）
        """
        self.window_size = window_size
        
        # 训练历史
        self.training_history = {
            'rounds': [],
            'losses': [],
            'f1_scores': [],
            'accuracies': [],
            'node_contributions': defaultdict(list),
            'parameter_changes': [],
            'communication_costs': []
        }
        
        # 收敛指标
        self.convergence_metrics = {
            'is_converged': False,
            'convergence_round': None,
            'convergence_rate': 0.0,
            'stability': 0.0
        }
    
    def record_round(
        self,
        round_num: int,
        loss: float,
        f1_score: float,
        accuracy: float,
        node_contributions: Optional[Dict[str, float]] = None,
        parameter_change: Optional[float] = None,
        communication_cost: Optional[float] = None
    ):
        """
        记录一轮训练的结果
        
        Args:
            round_num: 轮次编号
            loss: 损失值
            f1_score: F1分数
            accuracy: 准确率
            node_contributions: 节点贡献度字典
            parameter_change: 参数变化量
            communication_cost: 通信开销
        """
        self.training_history['rounds'].append(round_num)
        self.training_history['losses'].append(loss)
        self.training_history['f1_scores'].append(f1_score)
        self.training_history['accuracies'].append(accuracy)
        
        if node_contributions:
            for node_id, contribution in node_contributions.items():
                self.training_history['node_contributions'][node_id].append(contribution)
        
        if parameter_change is not None:
            self.training_history['parameter_changes'].append(parameter_change)
        
        if communication_cost is not None:
            self.training_history['communication_costs'].append(communication_cost)
        
        # 更新收敛指标
        self._update_convergence_metrics()
    
    def _update_convergence_metrics(self):
        """更新收敛指标"""
        losses = self.training_history['losses']
        
        if len(losses) < self.window_size:
            return
        
        # 计算最近窗口内的损失变化
        recent_losses = losses[-self.window_size:]
        loss_change = abs(recent_losses[-1] - recent_losses[0])
        loss_std = np.std(recent_losses)
        
        # 判断是否收敛
        convergence_threshold = 0.001
        if loss_change < convergence_threshold and loss_std < convergence_threshold:
            if not self.convergence_metrics['is_converged']:
                self.convergence_metrics['is_converged'] = True
                self.convergence_metrics['convergence_round'] = len(losses) - 1
        
        # 计算收敛率（损失下降速度）
        if len(losses) >= 2:
            loss_decrease = losses[0] - losses[-1]
            total_rounds = len(losses)
            self.convergence_metrics['convergence_rate'] = loss_decrease / total_rounds if total_rounds > 0 else 0
        
        # 计算稳定性（损失的标准差）
        self.convergence_metrics['stability'] = loss_std
    
    def check_convergence(self) -> Dict[str, Any]:
        """
        检查是否收敛
        
        Returns:
            收敛状态信息
        """
        return self.convergence_metrics.copy()
    
    def analyze_node_contributions(self) -> Dict[str, Any]:
        """
        分析节点贡献度
        
        Returns:
            节点贡献度分析结果
        """
        contributions = self.training_history['node_contributions']
        
        if not contributions:
            return {}
        
        analysis = {}
        for node_id, contrib_list in contributions.items():
            if contrib_list:
                analysis[node_id] = {
                    'average': np.mean(contrib_list),
                    'std': np.std(contrib_list),
                    'total': sum(contrib_list),
                    'trend': 'increasing' if contrib_list[-1] > contrib_list[0] else 'decreasing'
                }
        
        # 计算贡献度排名
        avg_contributions = {node_id: analysis[node_id]['average'] for node_id in analysis}
        sorted_nodes = sorted(avg_contributions.items(), key=lambda x: x[1], reverse=True)
        
        analysis['ranking'] = [node_id for node_id, _ in sorted_nodes]
        
        return analysis
    
    def calculate_parameter_change(
        self,
        old_params: Dict[str, torch.Tensor],
        new_params: Dict[str, torch.Tensor]
    ) -> float:
        """
        计算参数变化量
        
        Args:
            old_params: 旧参数
            new_params: 新参数
            
        Returns:
            参数变化量（L2范数）
        """
        total_change = 0.0
        
        for key in old_params.keys():
            if key in new_params:
                diff = new_params[key] - old_params[key]
                total_change += torch.norm(diff).item() ** 2
        
        return np.sqrt(total_change)
    
    def plot_convergence_curves(
        self,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        绘制收敛曲线
        
        Args:
            save_path: 保存路径
            show: 是否显示
        """
        rounds = self.training_history['rounds']
        losses = self.training_history['losses']
        f1_scores = self.training_history['f1_scores']
        accuracies = self.training_history['accuracies']
        
        if not rounds:
            print("没有数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 损失曲线
        axes[0, 0].plot(rounds, losses, 'b-', label='Loss')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # F1分数曲线
        axes[0, 1].plot(rounds, f1_scores, 'g-', label='F1 Score')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # 准确率曲线
        axes[1, 0].plot(rounds, accuracies, 'r-', label='Accuracy')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Accuracy')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # 参数变化曲线
        if self.training_history['parameter_changes']:
            param_changes = self.training_history['parameter_changes']
            param_rounds = rounds[:len(param_changes)]
            axes[1, 1].plot(param_rounds, param_changes, 'm-', label='Parameter Change')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Parameter Change (L2 Norm)')
            axes[1, 1].set_title('Parameter Changes')
            axes[1, 1].grid(True)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"收敛曲线已保存到: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_node_contributions(
        self,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        绘制节点贡献度图
        
        Args:
            save_path: 保存路径
            show: 是否显示
        """
        contributions = self.training_history['node_contributions']
        
        if not contributions:
            print("没有节点贡献度数据")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        rounds = self.training_history['rounds']
        
        # 时间序列图
        for node_id, contrib_list in contributions.items():
            contrib_rounds = rounds[:len(contrib_list)]
            axes[0].plot(contrib_rounds, contrib_list, label=f'Node {node_id}', marker='o')
        
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Contribution')
        axes[0].set_title('Node Contributions Over Time')
        axes[0].grid(True)
        axes[0].legend()
        
        # 平均贡献度柱状图
        avg_contributions = {
            node_id: np.mean(contrib_list)
            for node_id, contrib_list in contributions.items()
        }
        
        nodes = list(avg_contributions.keys())
        values = list(avg_contributions.values())
        
        axes[1].bar(nodes, values)
        axes[1].set_xlabel('Node ID')
        axes[1].set_ylabel('Average Contribution')
        axes[1].set_title('Average Node Contributions')
        axes[1].grid(True, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"节点贡献度图已保存到: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_convergence_report(self) -> Dict[str, Any]:
        """
        生成收敛报告
        
        Returns:
            收敛报告
        """
        report = {
            'convergence_status': self.convergence_metrics.copy(),
            'training_summary': {
                'total_rounds': len(self.training_history['rounds']),
                'final_loss': self.training_history['losses'][-1] if self.training_history['losses'] else None,
                'final_f1': self.training_history['f1_scores'][-1] if self.training_history['f1_scores'] else None,
                'final_accuracy': self.training_history['accuracies'][-1] if self.training_history['accuracies'] else None
            },
            'node_contributions': self.analyze_node_contributions(),
            'communication_efficiency': {
                'total_communication_cost': sum(self.training_history['communication_costs']) if self.training_history['communication_costs'] else None,
                'avg_communication_cost': np.mean(self.training_history['communication_costs']) if self.training_history['communication_costs'] else None
            }
        }
        
        return report
    
    def save_history(self, save_path: str):
        """
        保存训练历史
        
        Args:
            save_path: 保存路径
        """
        import json
        
        # 转换numpy类型为Python原生类型
        history = {}
        for key, value in self.training_history.items():
            if isinstance(value, list):
                history[key] = [float(v) if isinstance(v, (np.number, np.floating)) else v for v in value]
            elif isinstance(value, dict):
                history[key] = {
                    k: [float(v) if isinstance(v, (np.number, np.floating)) else v for v in val]
                    for k, val in value.items()
                }
            else:
                history[key] = value
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"训练历史已保存到: {save_path}")
