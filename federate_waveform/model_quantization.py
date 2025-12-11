# -*- coding: utf-8 -*-
"""
模型量化模块
支持动态量化和静态量化，用于模型压缩和加速
"""

import torch
import torch.nn as nn
import torch.quantization
from typing import Optional, Dict, Any, Tuple
import os


class ModelQuantizer:
    """模型量化器"""
    
    def __init__(self):
        self.quantization_configs = {
            'dynamic': self._dynamic_quantization,
            'static': self._static_quantization,
            'qat': self._quantization_aware_training
        }
    
    def quantize_model(
        self,
        model: nn.Module,
        quantization_type: str = 'dynamic',
        calibration_data: Optional[Any] = None,
        backend: str = 'fbgemm'
    ) -> nn.Module:
        """
        量化模型
        
        Args:
            model: 要量化的模型
            quantization_type: 量化类型 ('dynamic', 'static', 'qat')
            calibration_data: 校准数据（用于静态量化）
            backend: 量化后端 ('fbgemm', 'qnnpack')
            
        Returns:
            量化后的模型
        """
        if quantization_type not in self.quantization_configs:
            raise ValueError(f"不支持的量化类型: {quantization_type}")
        
        quantizer_func = self.quantization_configs[quantization_type]
        return quantizer_func(model, calibration_data, backend)
    
    def _dynamic_quantization(
        self,
        model: nn.Module,
        calibration_data: Optional[Any],
        backend: str
    ) -> nn.Module:
        """
        动态量化
        
        Args:
            model: 要量化的模型
            calibration_data: 未使用（动态量化不需要）
            backend: 量化后端
            
        Returns:
            量化后的模型
        """
        # 动态量化：只量化权重，激活在推理时量化
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},  # 要量化的层类型
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def _static_quantization(
        self,
        model: nn.Module,
        calibration_data: Optional[Any],
        backend: str
    ) -> nn.Module:
        """
        静态量化
        
        Args:
            model: 要量化的模型
            calibration_data: 校准数据（用于确定量化参数）
            backend: 量化后端
            
        Returns:
            量化后的模型
        """
        # 设置模型为评估模式
        model.eval()
        
        # 设置量化配置
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # 准备量化
        torch.quantization.prepare(model, inplace=True)
        
        # 使用校准数据校准
        if calibration_data is not None:
            print("正在进行校准...")
            with torch.no_grad():
                for data in calibration_data:
                    if isinstance(data, (tuple, list)):
                        model(data[0])
                    else:
                        model(data)
        
        # 转换为量化模型
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        return quantized_model
    
    def _quantization_aware_training(
        self,
        model: nn.Module,
        calibration_data: Optional[Any],
        backend: str
    ) -> nn.Module:
        """
        量化感知训练（QAT）
        注意：这需要在训练前设置，而不是训练后
        
        Args:
            model: 要量化的模型
            calibration_data: 未使用
            backend: 量化后端
            
        Returns:
            准备QAT的模型
        """
        # 设置QAT配置
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        
        # 准备QAT
        torch.quantization.prepare_qat(model, inplace=True)
        
        return model
    
    def save_quantized_model(
        self,
        model: nn.Module,
        save_path: str,
        quantization_type: str
    ):
        """
        保存量化模型
        
        Args:
            model: 量化后的模型
            save_path: 保存路径
            quantization_type: 量化类型
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # 保存量化模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'quantization_type': quantization_type,
            'model': model  # 保存完整模型（量化模型需要完整保存）
        }, save_path)
        
        print(f"量化模型已保存到: {save_path}")
    
    def load_quantized_model(
        self,
        load_path: str,
        model_class: Optional[type] = None
    ) -> nn.Module:
        """
        加载量化模型
        
        Args:
            load_path: 模型路径
            model_class: 模型类（如果需要）
            
        Returns:
            量化模型
        """
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
        
        if 'model' in checkpoint:
            # 如果保存了完整模型
            return checkpoint['model']
        elif model_class is not None:
            # 如果需要重建模型
            model = model_class()
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        else:
            raise ValueError("无法加载模型：需要提供model_class或保存完整模型")
    
    def compare_model_sizes(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module
    ) -> Dict[str, Any]:
        """
        比较原始模型和量化模型的大小
        
        Args:
            original_model: 原始模型
            quantized_model: 量化模型
            
        Returns:
            比较结果
        """
        # 计算模型大小
        def get_model_size(model):
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return param_size + buffer_size
        
        original_size = get_model_size(original_model)
        quantized_size = get_model_size(quantized_model)
        
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
        size_reduction = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
        
        return {
            'original_size_mb': original_size / (1024 ** 2),
            'quantized_size_mb': quantized_size / (1024 ** 2),
            'compression_ratio': compression_ratio,
            'size_reduction_percent': size_reduction
        }
    
    def evaluate_quantization_impact(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_data: Any,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        评估量化对模型性能的影响
        
        Args:
            original_model: 原始模型
            quantized_model: 量化模型
            test_data: 测试数据
            device: 设备
            
        Returns:
            评估结果
        """
        original_model.eval()
        quantized_model.eval()
        
        # 推理时间比较
        import time
        
        # 原始模型推理时间
        start = time.time()
        with torch.no_grad():
            for data in test_data:
                if isinstance(data, (tuple, list)):
                    _ = original_model(data[0].to(device))
                else:
                    _ = original_model(data.to(device))
        original_time = time.time() - start
        
        # 量化模型推理时间
        start = time.time()
        with torch.no_grad():
            for data in test_data:
                if isinstance(data, (tuple, list)):
                    _ = quantized_model(data[0].to(device))
                else:
                    _ = quantized_model(data.to(device))
        quantized_time = time.time() - start
        
        speedup = original_time / quantized_time if quantized_time > 0 else 0
        
        return {
            'original_inference_time': original_time,
            'quantized_inference_time': quantized_time,
            'speedup': speedup,
            'time_reduction_percent': (1 - quantized_time / original_time) * 100 if original_time > 0 else 0
        }
