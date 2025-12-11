# -*- coding: utf-8 -*-
"""
模型导出模块
支持导出为ONNX和TensorFlow Lite格式
"""

import torch
import torch.onnx
import os
from typing import Optional, Dict, Any, Tuple
import warnings


class ModelExporter:
    """模型导出器"""
    
    def __init__(self):
        self.supported_formats = ['onnx', 'tflite', 'torchscript']
    
    def export_to_onnx(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        input_names: Optional[list] = None,
        output_names: Optional[list] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 11,
        do_constant_folding: bool = True
    ) -> str:
        """
        导出模型为ONNX格式
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状 (batch, channels, height, width)
            output_path: 输出路径
            input_names: 输入名称列表
            output_names: 输出名称列表
            dynamic_axes: 动态轴配置
            opset_version: ONNX opset版本
            do_constant_folding: 是否进行常量折叠优化
            
        Returns:
            导出文件路径
        """
        model.eval()
        
        # 创建示例输入
        dummy_input = torch.randn(*input_shape)
        
        # 默认输入输出名称
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        try:
            # 导出ONNX模型
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                verbose=False
            )
            
            print(f"✅ ONNX模型已成功导出到: {output_path}")
            
            # 验证导出的模型
            self._verify_onnx_model(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"❌ ONNX导出失败: {e}")
            raise
    
    def _verify_onnx_model(self, onnx_path: str):
        """
        验证ONNX模型
        
        Args:
            onnx_path: ONNX模型路径
        """
        try:
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            print("✅ ONNX模型验证通过")
        except ImportError:
            warnings.warn("onnx包未安装，跳过验证。安装: pip install onnx")
        except Exception as e:
            warnings.warn(f"ONNX模型验证失败: {e}")
    
    def export_to_torchscript(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        method: str = 'trace'
    ) -> str:
        """
        导出模型为TorchScript格式
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状
            output_path: 输出路径
            method: 导出方法 ('trace' 或 'script')
            
        Returns:
            导出文件路径
        """
        model.eval()
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        try:
            if method == 'trace':
                # 使用追踪方法
                dummy_input = torch.randn(*input_shape)
                traced_model = torch.jit.trace(model, dummy_input)
                traced_model.save(output_path)
            elif method == 'script':
                # 使用脚本方法
                scripted_model = torch.jit.script(model)
                scripted_model.save(output_path)
            else:
                raise ValueError(f"不支持的导出方法: {method}")
            
            print(f"✅ TorchScript模型已成功导出到: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ TorchScript导出失败: {e}")
            raise
    
    def export_to_tflite(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        onnx_path: Optional[str] = None
    ) -> str:
        """
        导出模型为TensorFlow Lite格式
        注意：这需要先将PyTorch模型转换为ONNX，然后转换为TFLite
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状
            output_path: 输出路径
            onnx_path: 中间ONNX文件路径（如果已存在）
            
        Returns:
            导出文件路径
        """
        # 首先导出为ONNX
        if onnx_path is None:
            onnx_path = output_path.replace('.tflite', '.onnx')
            self.export_to_onnx(model, input_shape, onnx_path)
        
        # 将ONNX转换为TFLite
        try:
            # 这需要onnx-tf和tensorflow
            # 由于依赖复杂，这里提供转换指南
            print("=" * 60)
            print("TFLite转换指南:")
            print("=" * 60)
            print("1. 安装依赖:")
            print("   pip install onnx-tf tensorflow")
            print("")
            print("2. 使用以下代码转换:")
            print("   from onnx_tf.backend import prepare")
            print("   import tensorflow as tf")
            print("")
            print("   # 加载ONNX模型")
            print("   onnx_model = onnx.load(onnx_path)")
            print("   tf_rep = prepare(onnx_model)")
            print("")
            print("   # 转换为TFLite")
            print("   converter = tf.lite.TFLiteConverter.from_saved_model(tf_rep.export())")
            print("   tflite_model = converter.convert()")
            print("")
            print("   # 保存")
            print(f"   with open('{output_path}', 'wb') as f:")
            print("       f.write(tflite_model)")
            print("=" * 60)
            
            # 尝试自动转换（如果依赖可用）
            try:
                import onnx
                from onnx_tf.backend import prepare
                import tensorflow as tf
                
                # 加载ONNX模型
                onnx_model = onnx.load(onnx_path)
                tf_rep = prepare(onnx_model)
                
                # 转换为TFLite
                converter = tf.lite.TFLiteConverter.from_saved_model(tf_rep.export())
                tflite_model = converter.convert()
                
                # 保存
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(tflite_model)
                
                print(f"✅ TFLite模型已成功导出到: {output_path}")
                return output_path
                
            except ImportError:
                print("⚠️  缺少依赖，请按照上述指南手动转换")
                return None
                
        except Exception as e:
            print(f"❌ TFLite导出失败: {e}")
            print("请按照上述指南手动转换")
            raise
    
    def export_all_formats(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        output_dir: str,
        model_name: str = 'model'
    ) -> Dict[str, str]:
        """
        导出所有支持的格式
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状
            output_dir: 输出目录
            model_name: 模型名称
            
        Returns:
            导出文件路径字典
        """
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        # 导出ONNX
        onnx_path = os.path.join(output_dir, f'{model_name}.onnx')
        try:
            self.export_to_onnx(model, input_shape, onnx_path)
            exported_files['onnx'] = onnx_path
        except Exception as e:
            print(f"ONNX导出失败: {e}")
        
        # 导出TorchScript
        torchscript_path = os.path.join(output_dir, f'{model_name}.pt')
        try:
            self.export_to_torchscript(model, input_shape, torchscript_path)
            exported_files['torchscript'] = torchscript_path
        except Exception as e:
            print(f"TorchScript导出失败: {e}")
        
        # 导出TFLite（如果ONNX导出成功）
        if 'onnx' in exported_files:
            tflite_path = os.path.join(output_dir, f'{model_name}.tflite')
            try:
                self.export_to_tflite(model, input_shape, tflite_path, onnx_path)
                if tflite_path:
                    exported_files['tflite'] = tflite_path
            except Exception as e:
                print(f"TFLite导出失败: {e}")
        
        return exported_files
    
    def get_model_info(self, model: torch.nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model: 模型
            input_shape: 输入形状
            
        Returns:
            模型信息
        """
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        def get_model_size(model):
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return param_size + buffer_size
        
        model.eval()
        dummy_input = torch.randn(*input_shape)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        return {
            'input_shape': input_shape,
            'output_shape': output[0].shape if isinstance(output, tuple) else output.shape,
            'parameters': count_parameters(model),
            'model_size_mb': get_model_size(model) / (1024 ** 2),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
