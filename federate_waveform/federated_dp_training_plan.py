# -*- coding: utf-8 -*-
"""
差分隐私联邦学习训练计划
在基础训练计划基础上添加差分隐私保护
"""

import torch
import torch.nn as nn
import numpy as np
from federated_hypotension_training_plan import HypotensionTrainingPlan


class DPHypotensionTrainingPlan(HypotensionTrainingPlan):
    """
    带差分隐私的联邦学习训练计划
    在训练过程中添加梯度裁剪和噪声注入以实现差分隐私
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 差分隐私参数
        self.dp_enabled = True
        self.clip_norm = 1.0  # 梯度裁剪阈值
        self.noise_scale = 0.01  # 噪声规模 (sigma)
        self.delta = 1e-5  # Delta值 (通常设为1/n，n为数据集大小)
        self.epsilon = 0.0  # Epsilon值，会在训练过程中累积
        self.privacy_budget_consumed = 0.0  # 已消耗的隐私预算
        
    def set_dp_params(self, clip_norm=1.0, noise_scale=0.01, delta=1e-5):
        """
        设置差分隐私参数
        
        Args:
            clip_norm: 梯度裁剪阈值
            noise_scale: 噪声规模
            delta: Delta值
        """
        self.clip_norm = clip_norm
        self.noise_scale = noise_scale
        self.delta = delta
    
    def training_step(self, data, target):
        """
        执行一个训练步骤，包含差分隐私保护
        
        Args:
            data: 输入数据批次
            target: 目标标签批次
            
        Returns:
            Loss值
        """
        # 执行标准训练步骤
        loss = super().training_step(data, target)
        
        # 如果启用差分隐私，添加梯度裁剪和噪声
        if self.dp_enabled:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model().parameters(), 
                max_norm=self.clip_norm
            )
            
            # 添加高斯噪声到梯度
            for param in self.model().parameters():
                if param.grad is not None:
                    # 计算噪声规模（基于批次大小和总数据量）
                    batch_size = data.size(0)
                    # 假设我们知道数据集大小（实际应该从dataset获取）
                    # 这里使用一个合理的估计值
                    noise_std = self.noise_scale * self.clip_norm / np.sqrt(batch_size)
                    noise = torch.normal(
                        0, 
                        noise_std, 
                        size=param.grad.shape,
                        device=param.grad.device
                    )
                    param.grad += noise
            
            # 更新隐私预算（简化计算）
            # 实际应该使用RDP或GDP计算器
            self._update_privacy_budget()
        
        return loss
    
    def _update_privacy_budget(self):
        """
        更新隐私预算消耗
        这是一个简化的实现，实际应该使用RDP或GDP计算器
        """
        # 简化的隐私预算更新
        # 实际实现应该考虑采样率、噪声规模、迭代次数等
        batch_size = 128  # 默认批次大小
        # 这里使用一个简化的公式
        epsilon_increment = self.noise_scale / (batch_size * self.clip_norm)
        self.privacy_budget_consumed += epsilon_increment
    
    def get_privacy_budget(self):
        """
        获取当前隐私预算消耗情况
        
        Returns:
            dict: 包含epsilon和delta的字典
        """
        return {
            'epsilon': self.privacy_budget_consumed,
            'delta': self.delta,
            'noise_scale': self.noise_scale,
            'clip_norm': self.clip_norm
        }
    
    def init_dependencies(self):
        """
        声明节点端需要的依赖
        """
        deps = super().init_dependencies()
        deps.append("import numpy as np")
        return deps
