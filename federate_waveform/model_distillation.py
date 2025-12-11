# -*- coding: utf-8 -*-
"""
知识蒸馏模块
使用教师模型训练学生模型，实现模型压缩
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import copy


class KnowledgeDistillation:
    """知识蒸馏"""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.7
    ):
        """
        初始化知识蒸馏
        
        Args:
            teacher_model: 教师模型（大模型）
            student_model: 学生模型（小模型）
            temperature: 温度参数（软化概率分布）
            alpha: 蒸馏损失和真实标签损失的权重
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # 设置教师模型为评估模式
        self.teacher_model.eval()
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        计算蒸馏损失
        
        Args:
            student_logits: 学生模型输出（logits）
            teacher_logits: 教师模型输出（logits）
            labels: 真实标签
            temperature: 温度参数
            
        Returns:
            蒸馏损失
        """
        if temperature is None:
            temperature = self.temperature
        
        # 软化概率分布
        student_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        
        # KL散度损失（蒸馏损失）
        distillation_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # 学生模型在真实标签上的损失
        student_loss = F.cross_entropy(student_logits, labels)
        
        # 组合损失
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return total_loss
    
    def train_step(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        执行一个训练步骤
        
        Args:
            data: 输入数据
            labels: 标签
            optimizer: 优化器
            
        Returns:
            损失字典
        """
        self.student_model.train()
        optimizer.zero_grad()
        
        # 学生模型前向传播
        student_logits = self.student_model(data)
        
        # 教师模型前向传播（不计算梯度）
        with torch.no_grad():
            teacher_logits = self.teacher_model(data)
        
        # 计算蒸馏损失
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        with torch.no_grad():
            preds = torch.argmax(student_logits, dim=1)
            accuracy = (preds == labels).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
    
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        device: str = 'cpu'
    ) -> Dict[str, float]:
        """
        评估学生模型
        
        Args:
            data_loader: 数据加载器
            device: 设备
            
        Returns:
            评估指标
        """
        self.student_model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(device)
                labels = labels.to(device)
                
                # 学生模型输出
                student_logits = self.student_model(data)
                
                # 教师模型输出
                teacher_logits = self.teacher_model(data)
                
                # 计算损失
                loss = self.distillation_loss(student_logits, teacher_logits, labels)
                total_loss += loss.item()
                
                # 计算准确率
                preds = torch.argmax(student_logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    def compare_models(
        self,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        比较教师模型和学生模型的性能
        
        Args:
            test_data: 测试数据
            test_labels: 测试标签
            device: 设备
            
        Returns:
            比较结果
        """
        self.teacher_model.eval()
        self.student_model.eval()
        
        test_data = test_data.to(device)
        test_labels = test_labels.to(device)
        
        with torch.no_grad():
            # 教师模型预测
            teacher_logits = self.teacher_model(test_data)
            teacher_preds = torch.argmax(teacher_logits, dim=1)
            teacher_acc = (teacher_preds == test_labels).float().mean().item()
            
            # 学生模型预测
            student_logits = self.student_model(test_data)
            student_preds = torch.argmax(student_logits, dim=1)
            student_acc = (student_preds == test_labels).float().mean().item()
        
        # 计算模型大小
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        teacher_params = count_parameters(self.teacher_model)
        student_params = count_parameters(self.student_model)
        
        return {
            'teacher_accuracy': teacher_acc,
            'student_accuracy': student_acc,
            'accuracy_gap': teacher_acc - student_acc,
            'teacher_parameters': teacher_params,
            'student_parameters': student_params,
            'compression_ratio': teacher_params / student_params if student_params > 0 else 0,
            'size_reduction_percent': (1 - student_params / teacher_params) * 100 if teacher_params > 0 else 0
        }


def create_student_model(teacher_model: nn.Module, reduction_factor: float = 0.5) -> nn.Module:
    """
    根据教师模型创建学生模型（简化版本）
    
    Args:
        teacher_model: 教师模型
        reduction_factor: 模型大小缩减因子
        
    Returns:
        学生模型
    """
    # 这是一个示例实现，实际应该根据具体模型架构设计学生模型
    # 这里返回一个简化版本的模型
    
    # 获取教师模型的配置
    # 假设教师模型有特定的结构，我们需要创建一个更小的版本
    
    # 这里只是示例，实际实现需要根据具体模型架构
    student_model = copy.deepcopy(teacher_model)
    
    # 简化模型：减少层数或通道数
    # 实际实现应该更复杂，需要根据具体架构调整
    
    return student_model
