# -*- coding: utf-8 -*-
"""
模型剪枝模块
支持结构化剪枝和非结构化剪枝
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Tuple, Dict, Any, Optional
import copy


class ModelPruner:
    """模型剪枝器"""
    
    def __init__(self):
        self.pruning_methods = {
            'l1_unstructured': prune.L1Unstructured,
            'l2_unstructured': prune.L2Unstructured,
            'random_unstructured': prune.RandomUnstructured,
            'ln_structured': prune.LnStructured,
            'random_structured': prune.RandomStructured
        }
    
    def prune_model(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.3,
        pruning_type: str = 'l1_unstructured',
        layer_types: Optional[List[type]] = None
    ) -> nn.Module:
        """
        剪枝模型
        
        Args:
            model: 要剪枝的模型
            pruning_ratio: 剪枝比例 (0.0-1.0)
            pruning_type: 剪枝类型
            layer_types: 要剪枝的层类型列表，None则使用默认
            
        Returns:
            剪枝后的模型
        """
        if layer_types is None:
            layer_types = [nn.Linear, nn.Conv2d]
        
        if pruning_type not in self.pruning_methods:
            raise ValueError(f"不支持的剪枝类型: {pruning_type}")
        
        pruning_method = self.pruning_methods[pruning_type]
        
        # 收集要剪枝的参数
        parameters_to_prune = []
        for module in model.modules():
            if isinstance(module, tuple(layer_types)):
                parameters_to_prune.append((module, 'weight'))
        
        if not parameters_to_prune:
            print("警告: 没有找到可剪枝的层")
            return model
        
        # 执行全局剪枝
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=pruning_method,
            amount=pruning_ratio,
        )
        
        # 移除剪枝掩码，使剪枝永久化
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    def structured_prune(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.3,
        dim: int = 0,
        n: int = 2
    ) -> nn.Module:
        """
        结构化剪枝（按通道剪枝）
        
        Args:
            model: 要剪枝的模型
            pruning_ratio: 剪枝比例
            dim: 剪枝维度
            n: Ln范数的n值
            
        Returns:
            剪枝后的模型
        """
        parameters_to_prune = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        if not parameters_to_prune:
            print("警告: 没有找到可剪枝的层")
            return model
        
        # 使用Ln结构化剪枝
        prune.global_structured(
            parameters_to_prune,
            pruning_method=prune.LnStructured,
            amount=pruning_ratio,
            dim=dim,
            n=n
        )
        
        # 移除剪枝掩码
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    def iterative_pruning(
        self,
        model: nn.Module,
        total_pruning_ratio: float = 0.5,
        num_iterations: int = 5,
        pruning_type: str = 'l1_unstructured'
    ) -> nn.Module:
        """
        迭代剪枝（逐步剪枝，通常效果更好）
        
        Args:
            model: 要剪枝的模型
            total_pruning_ratio: 总剪枝比例
            num_iterations: 迭代次数
            pruning_type: 剪枝类型
            
        Returns:
            剪枝后的模型
        """
        pruning_ratio_per_iter = total_pruning_ratio / num_iterations
        
        for i in range(num_iterations):
            print(f"迭代剪枝 {i+1}/{num_iterations}, 当前剪枝比例: {(i+1) * pruning_ratio_per_iter:.2%}")
            model = self.prune_model(
                model,
                pruning_ratio=pruning_ratio_per_iter,
                pruning_type=pruning_type
            )
        
        return model
    
    def calculate_sparsity(self, model: nn.Module) -> Dict[str, float]:
        """
        计算模型稀疏度
        
        Args:
            model: 模型
            
        Returns:
            稀疏度信息
        """
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0.0
        
        return {
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'sparsity': sparsity,
            'sparsity_percent': sparsity * 100
        }
    
    def get_pruning_statistics(
        self,
        original_model: nn.Module,
        pruned_model: nn.Module
    ) -> Dict[str, Any]:
        """
        获取剪枝统计信息
        
        Args:
            original_model: 原始模型
            pruned_model: 剪枝后的模型
            
        Returns:
            统计信息
        """
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        def count_nonzero_parameters(model):
            return sum((p != 0).sum().item() for p in model.parameters())
        
        original_params = count_parameters(original_model)
        pruned_params = count_parameters(pruned_model)
        pruned_nonzero = count_nonzero_parameters(pruned_model)
        
        reduction = (1 - pruned_params / original_params) * 100 if original_params > 0 else 0
        
        return {
            'original_parameters': original_params,
            'pruned_parameters': pruned_params,
            'nonzero_parameters': pruned_nonzero,
            'parameter_reduction_percent': reduction,
            'sparsity': self.calculate_sparsity(pruned_model)['sparsity']
        }
    
    def save_pruned_model(
        self,
        model: nn.Module,
        save_path: str,
        pruning_info: Optional[Dict[str, Any]] = None
    ):
        """
        保存剪枝后的模型
        
        Args:
            model: 剪枝后的模型
            save_path: 保存路径
            pruning_info: 剪枝信息
        """
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'pruning_info': pruning_info or {}
        }, save_path)
        
        print(f"剪枝模型已保存到: {save_path}")
