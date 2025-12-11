# 联邦波形分类系统 (Federated Waveform Classification)

基于自研联邦学习框架的波形分类系统，支持低血压预测等医疗场景应用。

## 环境配置

### 1. 创建虚拟环境

```bash
# 进入项目目录
cd your_path_to/federate_waveform

# 创建虚拟环境（习惯用conda可以自己换）
python3 -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### 2. 安装依赖

```bash
# 安装所有必需依赖
pip install torch torchvision numpy pandas scipy pyyaml
pip install Flask==2.3.3 flask-socketio==5.3.5 eventlet==0.33.3
pip install psutil matplotlib scikit-learn

# 可选：模型导出
pip install onnx
```

## 快速开始

### 1. 数据准备

```bash
cd federate_waveform
python prepare_federated_data.py
```

确保 `uci2_dataset/` 目录在项目根目录下，包含 `feat_fold_0.csv`, `feat_fold_1.csv`, `feat_fold_2.csv` 文件。

### 2. 启动训练

#### 方式一：可视化界面（推荐）

```bash
python federated_learning_visualization.py
```

访问 `http://localhost:5002`，在界面中：
- 激活节点（node_1, node_2, node_3）
- 配置训练参数
- 点击"开始训练"

#### 方式二：命令行

```bash
# 基础实验
python federated_hypotension_experiment.py

# 带监控的实验
python federated_hypotension_experiment_with_monitor.py
```

### 3. 查看结果

训练结果自动保存到 `results/` 目录：
- `results/models/best_model.pth` - 最佳模型
- `results/training_data/training_history.json` - 训练历史
- `results/training_data/round_N.json` - 每轮详细数据

## 配置说明

### 设备配置 (`devices.yaml`)

可以配置每个节点的硬件参数和在线模式：
- `online_pattern`: always_on, mostly_online, intermittent, sporadic
- `compute_power`: low, medium, high

### 训练配置

在可视化界面或代码中配置：
- `rounds`: 训练轮数（默认5）
- `batch_size`: 批次大小（默认128）
- `learning_rate`: 学习率（默认4e-5）

## 故障排除

1. **数据路径错误**: 确保 `uci2_dataset/` 目录存在且包含CSV文件
2. **端口被占用**: 修改 `federated_learning_visualization.py` 中的端口号（默认5002）
3. **节点未参与**: 检查节点是否已激活，检查 `devices.yaml` 配置

## 目录结构

```
federate_waveform/
├── federated_simulation_trainer.py        # 联邦训练核心模块
├── federated_learning_visualization.py    # 可视化监控系统
├── prepare_federated_data.py              # 数据准备脚本
├── devices.yaml                           # 设备配置文件
└── ...
```

---

**版本**: v2.0 (自研联邦学习框架版本)
