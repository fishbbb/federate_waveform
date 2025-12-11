# -*- coding: utf-8 -*-
"""
Fed-BioMed Training Plan for Hypotension Prediction
This file integrates the ECGNet model with Fed-BioMed framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy import signal
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.data import DataManager
from torch.utils.data import Dataset
import os
import sys
import time
import random
import yaml

# Add the Waveform-classification program directory to path
# 从federate_waveform文件夹向上两级到根目录，然后访问Waveform-classification
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
waveform_program_dir = os.path.join(base_dir, 'Waveform-classification', 'program')
sys.path.insert(0, waveform_program_dir)

# Import model from the original project
from models import myecgnet
from utils import Loss_cal

# ===== 设备选择函数：优先 CUDA（Windows / Linux），然后 MPS（macOS M 系列） =====
def get_device():
    """
    选择计算设备：优先 CUDA（Windows / Linux），然后 MPS（macOS M 系列），最后 CPU
    """
    # Windows / Linux 上如果有 NVIDIA GPU：
    if torch.cuda.is_available():
        return torch.device("cuda")  # Windows 环境下建议这样写

    # macOS Apple Silicon 上优先 MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # 如果你将来在 Windows 上没有 GPU，也会回退到 CPU
    return torch.device("cpu")

DEVICE = get_device()
print(f"[Simulation] Using device: {DEVICE}")

# ===== 从 devices.yaml 读取设备配置 =====
def load_device_profiles(config_path=None):
    """
    加载设备配置文件
    
    Args:
        config_path: devices.yaml 文件路径，如果为None则自动查找
        
    Returns:
        设备配置列表
    """
    if config_path is None:
        # 尝试从当前文件所在目录查找
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "devices.yaml")
    
    if not os.path.exists(config_path):
        print(f"[Simulation] Warning: devices.yaml not found at {config_path}, using default profile")
        # 返回默认配置
        return [{
            "id": "node_1",
            "fb_node_path": "fbm-node-1",
            "type": "default",
            "compute_power": "medium",
            "cpu_threads": 2,
            "max_batch_size": 64,
            "simulated_compute_delay_ms": 0,
            "upload_latency_ms": 0,
            "download_latency_ms": 0,
            "bandwidth_kbps": 1000,
            "online_pattern": "always_on",
            "model_variant": "base"
        }]
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("devices", [])

# 全局加载设备配置（在模块加载时）
try:
    DEVICE_PROFILES = load_device_profiles()
except Exception as e:
    print(f"[Simulation] Error loading device profiles: {e}")
    DEVICE_PROFILES = []

def get_profile_for_fb_node(fb_node_path: str):
    """
    根据 Fed-BioMed node 的 path（比如 fbm-node-1）找到对应的设备配置
    
    Args:
        fb_node_path: Fed-BioMed 节点的路径标识
        
    Returns:
        设备配置字典
    """
    for d in DEVICE_PROFILES:
        if d.get("fb_node_path") == fb_node_path:
            return d
    
    # 如果找不到，返回默认配置
    print(f"[Simulation] Warning: No device profile found for fb_node_path = {fb_node_path}, using default")
    return {
        "id": fb_node_path.replace("fbm-node-", "node_"),
        "fb_node_path": fb_node_path,
        "type": "default",
        "compute_power": "medium",
        "cpu_threads": 2,
        "max_batch_size": 64,
        "simulated_compute_delay_ms": 0,
        "upload_latency_ms": 0,
        "download_latency_ms": 0,
        "bandwidth_kbps": 1000,
        "online_pattern": "always_on",
        "model_variant": "base"
    }

# ===== 仿真设备在线 / 掉线模式 =====
def is_available_this_round(profile, round_id: int):
    """
    判断设备在当前轮次是否可用
    
    Args:
        profile: 设备配置
        round_id: 当前轮次ID
        
    Returns:
        bool: 是否可用
    """
    pattern = profile.get("online_pattern", "always_on")
    r = random.random()

    if pattern == "always_on":
        return True
    elif pattern == "mostly_online":
        # 大约 80% 轮参与
        return r < 0.8
    elif pattern == "intermittent":
        # 大约 40% 轮参与
        return r < 0.4
    elif pattern == "sporadic":
        # 大约 20% 轮参与
        return r < 0.2
    else:
        # 默认：保守一点，当作间歇在线
        return r < 0.5

# ===== 在本地训练开始前，按 profile 配置资源 =====
def apply_compute_profile(profile):
    """
    应用计算资源配置
    
    Args:
        profile: 设备配置
        
    Returns:
        float: 每个 batch 的模拟计算延迟（秒）
    """
    cpu_threads = int(profile.get("cpu_threads", 2))
    torch.set_num_threads(cpu_threads)

    # Windows 上如果想限制 CPU 线程数，代码相同
    # torch.set_num_threads(cpu_threads)  # Windows / Linux 也是用这一行

    # 返回每个 batch 的"模拟计算延迟"
    delay_s = profile.get("simulated_compute_delay_ms", 0) / 1000.0
    return delay_s

# ===== 网络模拟：在向服务器发送更新前，sleep 模拟上传时间 =====
def simulate_network_delay(profile, model_size_bytes: int):
    """
    模拟网络延迟
    
    Args:
        profile: 设备配置
        model_size_bytes: 模型大小（字节）
        
    Returns:
        float: 总延迟时间（秒）
    """
    upload_latency = profile.get("upload_latency_ms", 0) / 1000.0
    bandwidth_kbps = profile.get("bandwidth_kbps", 1000)  # kilobit per second

    # 模型大小（byte） -> kbit
    model_size_kbit = (model_size_bytes * 8) / 1000.0
    # 传输时间 = 大小 / 带宽
    transfer_time = model_size_kbit / max(bandwidth_kbps, 1e-6)

    total_delay = upload_latency + transfer_time
    time.sleep(total_delay)
    return total_delay

def get_model_size_bytes(model_state_dict):
    """
    计算模型大小（字节）
    
    Args:
        model_state_dict: 模型状态字典
        
    Returns:
        int: 模型大小（字节）
    """
    total = 0
    for v in model_state_dict.values():
        total += v.numel() * v.element_size()
    return total


class HypotensionDataset(Dataset):
    """Dataset class for hypotension prediction data (supports uci2_dataset)"""
    
    def __init__(self, data_path, csv_path=None, uci2_base_dir=None, train=True, use_uci2=True):
        """
        Args:
            data_path: Path to the .pth file containing train/val split
            csv_path: Path to the CSV file with waveform data (for old format)
            uci2_base_dir: Base directory for uci2_dataset (for new format)
            train: Whether to use training or validation split
            use_uci2: If True, use uci2_dataset format; if False, use old CSV format
        """
        super(HypotensionDataset, self).__init__()
        
        # Load train/val split
        split_data = torch.load(data_path, map_location="cpu", weights_only=False)
        raw_ids = split_data["train"] if train else split_data["val"]
        self.id = [str(x) for x in raw_ids]
        self.target_point_num = 1000
        self.train = train
        self.use_uci2 = use_uci2
        
        if use_uci2:
            # 使用uci2_dataset格式
            if uci2_base_dir is None:
                # 尝试从data_path推断
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(data_path))))
                uci2_base_dir = os.path.join(base_dir, 'uci2_dataset')
            
            # 加载所有fold的数据
            self._load_uci2_data(uci2_base_dir, raw_ids)
        else:
            # 使用旧的CSV格式
            if csv_path is None:
                raise ValueError("csv_path must be provided when use_uci2=False")
            
            # Read CSV source data
            source_data = pd.read_csv(csv_path)
            
            # Set index to 'id' column
            if "id" in source_data.columns:
                source_data = source_data.set_index("id", drop=False)
            
            # Ensure consistent types
            source_data.index = source_data.index.astype(str)
            
            # Check for missing IDs
            missing = [x for x in self.id if x not in source_data.index]
            if missing:
                sample_miss = ", ".join(missing[:10])
                raise KeyError(
                    f"{len(missing)} ids not in DataFrame index. "
                    f"Examples: [{sample_miss}] ..."
                )
            
            # Extract waveform features (columns 4 to 1253 for old format)
            wave_features = [str(x) for x in range(4, 1254)]
            self.data = source_data.loc[self.id, wave_features].to_numpy()
            self.label = source_data.loc[self.id, 'label'].to_numpy()
    
    def _load_uci2_data(self, uci2_base_dir, raw_ids):
        """
        加载uci2_dataset数据
        
        Args:
            uci2_base_dir: uci2_dataset基础目录
            raw_ids: 要加载的ID列表
        """
        all_data = []
        all_labels = []
        found_ids = set()
        
        # 尝试从所有fold加载数据
        for fold in [0, 1, 2]:
            csv_path = os.path.join(uci2_base_dir, f'feat_fold_{fold}.csv')
            if not os.path.exists(csv_path):
                continue
            
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            
            # 创建ID
            df['id'] = df['patient'].astype(str) + '_' + df['trial'].astype(str)
            
            # 创建低血压标签
            df['label'] = ((df['SP'] < 90) | (df['DP'] < 60)).astype(int)
            
            # 设置索引
            df = df.set_index('id', drop=False)
            
            # 提取需要的ID
            needed_ids = [id_str for id_str in raw_ids if id_str in df.index and id_str not in found_ids]
            
            if needed_ids:
                # 获取特征列（排除patient, trial, SP, DP, id, label）
                feature_cols = [col for col in df.columns 
                               if col not in ['patient', 'trial', 'SP', 'DP', 'id', 'label']]
                
                # 提取特征和标签
                features = df.loc[needed_ids, feature_cols].values
                labels = df.loc[needed_ids, 'label'].values
                
                all_data.append(features)
                all_labels.append(labels)
                found_ids.update(needed_ids)
        
        if not all_data:
            raise ValueError(f"No data found for IDs in uci2_dataset at {uci2_base_dir}")
        
        # 合并所有数据
        self.data = np.vstack(all_data)
        self.label = np.hstack(all_labels)
        
        # 检查是否所有ID都找到了
        missing_ids = set(raw_ids) - found_ids
        if missing_ids:
            print(f"Warning: {len(missing_ids)} IDs not found in uci2_dataset")
            # 只保留找到的ID
            found_id_list = [id_str for id_str in raw_ids if id_str in found_ids]
            self.id = found_id_list
            # 重新索引数据
            id_to_idx = {id_str: idx for idx, id_str in enumerate(found_id_list)}
            indices = [id_to_idx[id_str] for id_str in found_id_list if id_str in found_ids]
            self.data = self.data[indices]
            self.label = self.label[indices]
        
        print(f"Loaded {len(self.data)} samples from uci2_dataset")
        print(f"  Features shape: {self.data.shape}")
        print(f"  Hypotension: {self.label.sum()} ({self.label.sum()/len(self.label)*100:.2f}%)")
    
    def resample(self, sig, target_point_num=None):
        """Resample signal to target length"""
        sig = signal.resample(sig, target_point_num) if target_point_num else sig
        return np.array(sig).reshape(-1, 1)
    
    def scaling(self, X, sigma=0.05):
        """Add scaling noise for data augmentation"""
        scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
        myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
        return X * myNoise
    
    def shift(self, sig, interval=10):
        """Add shift noise for data augmentation"""
        for col in range(sig.shape[1]):
            offset = np.random.choice(range(-interval, interval))
            sig[:, col] += offset
        return sig
    
    def transform(self, sig):
        """Transform signal with resampling and augmentation"""
        sig = self.resample(sig, self.target_point_num)
        np.random.seed()
        if self.train:
            if np.random.rand() > 0.5:
                sig = self.scaling(sig)
            if np.random.rand() > 0.5:
                sig = self.shift(sig)
        sig = sig.transpose()  # (2500,8) -> (8,2500)
        sig = torch.tensor(sig.copy(), dtype=torch.float32)  # 明确使用 float32（MPS 不支持 float64）
        return sig
    
    def transform_features(self, features):
        """
        将特征向量转换为目标长度的信号（用于uci2_dataset）
        
        Args:
            features: 特征向量（222维）
            
        Returns:
            转换后的信号（1000个点）
        """
        # 如果特征数量小于目标长度，进行填充
        if len(features) < self.target_point_num:
            # 使用重复填充
            repeat_times = self.target_point_num // len(features) + 1
            features_padded = np.tile(features, repeat_times)[:self.target_point_num]
        elif len(features) > self.target_point_num:
            # 如果特征数量大于目标长度，进行降采样
            features_padded = signal.resample(features, self.target_point_num)
        else:
            features_padded = features
        
        # 数据增强（仅在训练时）
        if self.train:
            np.random.seed()
            if np.random.rand() > 0.5:
                # 添加噪声
                noise = np.random.normal(0, 0.02, features_padded.shape)
                features_padded = features_padded + noise
        
        # 返回 numpy 数组（将在 __getitem__ 中转换为 float32 tensor）
        return features_padded.reshape(-1, 1)
    
    def __getitem__(self, index):
        features = self.data[index]
        
        if self.use_uci2:
            # 使用特征转换（uci2_dataset有222个特征）
            x = self.transform_features(features).reshape((1, 1, self.target_point_num))
            # 确保转换为 float32 tensor（MPS 不支持 float64）
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            elif not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            elif x.dtype != torch.float32:
                x = x.float()
        else:
            # 使用原始信号转换（旧格式）
            x = self.transform(features).reshape((1, 1, self.target_point_num))
            # 确保是 float32
            if x.dtype != torch.float32:
                x = x.float()
        
        target = self.label[index]
        target = torch.tensor(target, dtype=torch.float32)
        return x, target
    
    def __len__(self):
        return len(self.id)


class HypotensionTrainingPlan(TorchTrainingPlan):
    """
    Fed-BioMed Training Plan for Hypotension Prediction
    This class integrates the ECGNet model with Fed-BioMed framework
    """
    
    def __init__(self, *args, **kwargs):
        """
        初始化训练计划，加载设备配置
        """
        super().__init__(*args, **kwargs)
        
        # 1) 确定当前节点的 fb_node_path（可以从 env 里读）
        fb_node_path = os.environ.get("FB_NODE_PATH", None)
        if fb_node_path is None:
            # 尝试从 model_args 中获取
            model_args = kwargs.get('model_args', {})
            if isinstance(model_args, dict):
                fb_node_path = model_args.get('fb_node_path', 'fbm-node-1')
            else:
                fb_node_path = 'fbm-node-1'
        
        # 2) 根据 fb_node_path 载入 profile
        self.device_profile = get_profile_for_fb_node(fb_node_path)
        
        # 3) 选择 device（Mac 用 MPS，Windows 上可以在 get_device 里优先 CUDA）
        self.device = DEVICE
        
        # 4) 应用计算资源配置（限制CPU线程数）
        apply_compute_profile(self.device_profile)
        
        print(f"[Simulation] Initialized training plan for device: {self.device_profile['id']}")
        print(f"[Simulation] Device type: {self.device_profile['type']}, Compute power: {self.device_profile['compute_power']}")
        print(f"[Simulation] Using device: {self.device}")
        print(f"[Simulation] CPU threads: {self.device_profile.get('cpu_threads', 2)}, Max batch size: {self.device_profile.get('max_batch_size', 64)}")
    
    def init_model(self, model_args):
        """
        Initialize the ECGNet model
        
        Args:
            model_args: Dictionary of model arguments
            
        Returns:
            ECGNet model instance
        """
        return self.ECGNetModel(model_args=model_args)
    
    def init_optimizer(self, optimizer_args):
        """
        Initialize optimizer
        
        Args:
            optimizer_args: Dictionary with optimizer arguments (e.g., {'lr': 4e-5})
            
        Returns:
            Optimizer instance
        """
        return torch.optim.AdamW(
            self.model().parameters(),
            lr=optimizer_args.get('lr', 4e-5)
        )
    
    def init_dependencies(self):
        """
        Declare dependencies needed on the node side
        
        Returns:
            List of import statements as strings
        """
        deps = [
            "import torch",
            "import torch.nn as nn",
            "import numpy as np",
            "import pandas as pd",
            "from scipy import signal",
            "import math",
        ]
        return deps
    
    def training_data(self):
        """
        Load and return training data
        
        Returns:
            DataManager instance with the dataset
        """
        # Get paths from model_args or use defaults
        model_args = self.model_args()
        use_uci2 = model_args.get('use_uci2', True)  # 默认使用uci2_dataset
        
        if use_uci2:
            # 使用uci2_dataset
            # 重要：优先使用 dataset_path（由 Fed-BioMed 自动设置，指向节点实际数据位置）
            # model_args 中的 data_path 仅作为回退值
            default_data_path = model_args.get('data_path', 'federated_data/node_1/train.pth')
            uci2_base_dir = model_args.get('uci2_base_dir', 'uci2_dataset')
            
            # Resolve absolute paths if needed
            # 优先级：1) dataset_path（节点实际数据） 2) model_args 中的路径 3) 默认路径
            if hasattr(self, 'dataset_path') and self.dataset_path:
                # Fed-BioMed 自动设置的节点数据路径（最高优先级）
                # dataset_path 通常是节点上数据集的实际路径
                if os.path.exists(self.dataset_path):
                    # 如果 dataset_path 是目录，尝试在其中查找 train.pth
                    if os.path.isdir(self.dataset_path):
                        potential_path = os.path.join(self.dataset_path, 'train.pth')
                        if os.path.exists(potential_path):
                            data_path = potential_path
                        else:
                            # 如果目录中没有 train.pth，使用目录本身（可能包含数据文件）
                            data_path = self.dataset_path
                    else:
                        # dataset_path 是文件路径
                        data_path = self.dataset_path
                else:
                    # dataset_path 不存在，尝试使用 model_args 中的路径
                    data_path = default_data_path
            else:
                # 没有 dataset_path，使用 model_args 中的路径
                data_path = default_data_path
            
            # 如果 data_path 不是绝对路径，尝试解析
            if not os.path.isabs(data_path):
                # 尝试在 dataset_path 目录中查找
                if hasattr(self, 'dataset_path') and self.dataset_path and os.path.isdir(self.dataset_path):
                    potential_path = os.path.join(self.dataset_path, data_path)
                    if os.path.exists(potential_path):
                        data_path = potential_path
                # 尝试在 federated_data 目录中查找
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                federated_data_path = os.path.join(base_dir, data_path)
                if os.path.exists(federated_data_path):
                    data_path = federated_data_path
                # 最后尝试当前目录
                elif os.path.exists(data_path):
                    data_path = os.path.abspath(data_path)
                else:
                    raise FileNotFoundError(f"Cannot find data_path: {data_path}. "
                                          f"dataset_path={getattr(self, 'dataset_path', 'None')}")
            
            if not os.path.isabs(uci2_base_dir):
                # 优先级：1) dataset_path 目录 2) 项目根目录 3) 当前目录
                if hasattr(self, 'dataset_path') and self.dataset_path and os.path.isdir(self.dataset_path):
                    # 尝试在 dataset_path 的父目录中查找 uci2_dataset
                    dataset_parent = os.path.dirname(self.dataset_path)
                    potential_uci2 = os.path.join(dataset_parent, uci2_base_dir)
                    if os.path.exists(potential_uci2):
                        uci2_base_dir = potential_uci2
                # 尝试项目根目录
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                root_uci2_path = os.path.join(base_dir, uci2_base_dir)
                if os.path.exists(root_uci2_path):
                    uci2_base_dir = root_uci2_path
                # 最后尝试当前目录
                elif os.path.exists(uci2_base_dir):
                    uci2_base_dir = os.path.abspath(uci2_base_dir)
                else:
                    raise FileNotFoundError(f"Cannot find uci2_base_dir: {uci2_base_dir}. "
                                          f"dataset_path={getattr(self, 'dataset_path', 'None')}")
            
            # Create dataset
            dataset = HypotensionDataset(
                data_path=data_path,
                uci2_base_dir=uci2_base_dir,
                train=True,
                use_uci2=True
            )
        else:
            # 使用旧的CSV格式
            data_path = model_args.get('data_path', 'wide_data/train.pth')
            csv_path = model_args.get('csv_path', 'source_data/hypotensive_forecast_data_new(samples).csv')
            
            # Resolve absolute paths if needed
            if not os.path.isabs(data_path):
                if hasattr(self, 'dataset_path') and os.path.exists(os.path.join(self.dataset_path, data_path)):
                    data_path = os.path.join(self.dataset_path, data_path)
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                waveform_data_path = os.path.join(base_dir, 'Waveform-classification', data_path)
                if os.path.exists(waveform_data_path):
                    data_path = waveform_data_path
                elif os.path.exists(data_path):
                    data_path = os.path.abspath(data_path)
                else:
                    raise FileNotFoundError(f"Cannot find data_path: {data_path}")
            
            if not os.path.isabs(csv_path):
                if hasattr(self, 'dataset_path') and os.path.exists(os.path.join(self.dataset_path, csv_path)):
                    csv_path = os.path.join(self.dataset_path, csv_path)
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                waveform_csv_path = os.path.join(base_dir, 'Waveform-classification', csv_path)
                if os.path.exists(waveform_csv_path):
                    csv_path = waveform_csv_path
                elif hasattr(self, 'dataset_path') and os.path.exists(os.path.join(self.dataset_path, os.path.basename(csv_path))):
                    csv_path = os.path.join(self.dataset_path, os.path.basename(csv_path))
                elif os.path.exists(csv_path):
                    csv_path = os.path.abspath(csv_path)
                else:
                    raise FileNotFoundError(f"Cannot find csv_path: {csv_path}")
            
            # Create dataset
            dataset = HypotensionDataset(
                data_path=data_path,
                csv_path=csv_path,
                train=True,
                use_uci2=False
            )
        
        # Return DataManager with dataset
        # 根据设备配置调整批次大小
        default_batch_size = model_args.get('batch_size', 128)
        max_batch_size = self.device_profile.get('max_batch_size', default_batch_size) if hasattr(self, 'device_profile') else default_batch_size
        batch_size = min(default_batch_size, max_batch_size)
        
        train_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 0  # Set to 0 to avoid multiprocessing issues
        }
        
        return DataManager(dataset=dataset, **train_kwargs)
    
    def training_step(self, data, target):
        """
        Execute one training step with device simulation
        
        Args:
            data: Input data batch
            target: Target labels batch
            
        Returns:
            Loss value
        """
        # 确保设备配置已加载
        if not hasattr(self, 'device_profile'):
            # 如果还没有加载，尝试加载
            fb_node_path = os.environ.get("FB_NODE_PATH", 'fbm-node-1')
            self.device_profile = get_profile_for_fb_node(fb_node_path)
            self.device = DEVICE
        
        # 将数据移到正确的设备上
        if hasattr(self, 'device'):
            data = data.to(self.device)
            target = target.to(self.device)
        
        # Ensure target has correct shape
        target = target.view(-1, 1)
        
        # Forward pass
        output, feature = self.model().forward(data)
        
        # Calculate loss using focal loss
        criterion = Loss_cal(alpha=0.5, gamma=2.0)
        loss = criterion(output, target)
        
        # 模拟计算延迟（如果配置了）
        if hasattr(self, 'device_profile'):
            compute_delay = self.device_profile.get('simulated_compute_delay_ms', 0) / 1000.0
            if compute_delay > 0:
                time.sleep(compute_delay)
        
        return loss
    
    class ECGNetModel(nn.Module):
        """
        ECGNet model wrapper for Fed-BioMed
        This is a wrapper around the original myecgnet model
        Supports different model variants based on device profile
        """
        
        def __init__(self, model_args):
            super().__init__()
            # 根据设备配置选择模型变体（如果需要）
            # 目前所有设备使用相同的模型，但可以扩展为不同大小的模型
            model_variant = model_args.get('model_variant', 'base')
            
            # Initialize the original ECGNet model
            self.ecgnet = myecgnet(pretrained=False, num_classes=1)
            
            # 如果将来需要支持不同大小的模型，可以在这里实现
            # if model_variant == 'small':
            #     # 使用更小的模型
            #     self.ecgnet = create_small_model()
            # elif model_variant == 'large':
            #     # 使用更大的模型
            #     self.ecgnet = create_large_model()
        
        def forward(self, x):
            """
            Forward pass
            
            Args:
                x: Input tensor of shape (batch, 1, 1, 1000)
                
            Returns:
                output: Model output (logits)
                feature: Extracted features (for compatibility)
            """
            # The original model expects (batch, 1, 1, 1000)
            # Ensure correct shape
            if len(x.shape) == 3:
                x = x.unsqueeze(1)  # Add channel dimension if missing
            
            output, feature = self.ecgnet(x)
            return output, feature

