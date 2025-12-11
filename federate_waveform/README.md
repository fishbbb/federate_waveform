# 联邦波形分类系统 (Federated Waveform Classification)

基于自研联邦学习框架的波形分类系统，支持低血压预测等医疗场景应用。系统采用单进程模拟多客户端联邦训练，无需依赖 Fed-BioMed 框架。

## 项目概述

本项目实现了一个完整的联邦边缘智能系统，用于波形数据的分类任务。系统支持：

- **自研联邦学习训练**: 单进程模拟多节点协作训练，实现 FedAvg 聚合算法
- **硬件感知优化**: 设备性能分析和资源感知调度
- **设备在线模式模拟**: 支持 always_on、mostly_online、intermittent、sporadic 等在线模式
- **实时监控**: 可视化训练过程和收敛分析
- **实时数据保存**: 训练过程中自动保存每轮数据到 `results/` 文件夹
- **模型版本管理**: 每轮保存 current 模型，自动跟踪并保存 best 模型
- **模型压缩**: 集成量化、剪枝、知识蒸馏等模型压缩技术
- **模型导出**: 支持 ONNX 和 TensorFlow Lite 格式

## 目录结构

```
federate_waveform/
├── README.md                              # 本文件
├── federated_simulation_trainer.py        # 自研联邦训练核心模块（重要）
├── federated_learning_visualization.py    # 可视化监控系统（Flask + SocketIO）
├── federated_hypotension_training_plan.py # 训练计划（数据集、设备仿真工具函数）
├── federated_hypotension_experiment.py    # 基础实验脚本
├── federated_hypotension_experiment_with_monitor.py  # 带监控的实验脚本
├── prepare_federated_data.py              # 数据准备脚本
├── devices.yaml                           # 设备配置文件（硬件参数、在线模式）
├── hardware_profiling.py                  # 硬件分析模块
├── resource_aware_scheduler.py            # 资源感知调度器
├── model_quantization.py                  # 模型量化模块
├── model_pruning.py                       # 模型剪枝模块
├── model_distillation.py                  # 知识蒸馏模块
├── model_export.py                        # 模型导出模块
├── adaptive_system.py                     # 自适应系统模块
└── convergence_monitoring.py              # 收敛监控模块
```

## 环境配置

### 步骤 1：创建虚拟环境（推荐）

为了隔离项目依赖，避免与其他项目冲突，强烈建议使用虚拟环境。

#### 1.1 检查 Python 版本

首先确认系统已安装 Python 3.8 或更高版本（推荐 3.10）：

```bash
# 检查 Python 版本
python3 --version
# 或
python --version
```

如果未安装 Python 或版本过低，请先安装 Python 3.8+。

#### 1.2 创建虚拟环境

在项目根目录下创建虚拟环境：

```bash
# 进入项目目录
cd your_path_to/federate_waveform

# 创建虚拟环境（会在当前目录下创建 venv 文件夹），习惯用conda可以自己换
python3 -m venv venv
```

**注意**：如果 `python3` 命令不存在，可以尝试使用 `python` 命令。

#### 1.3 激活虚拟环境

根据你的操作系统，使用相应的命令激活虚拟环境：

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
# PowerShell
venv\Scripts\Activate.ps1

# 或 CMD
venv\Scripts\activate.bat
```

激活成功后，命令行提示符前会显示 `(venv)` 标识，例如：
```bash
(venv) user@hostname federate_waveform %
```

#### 1.4 验证虚拟环境

确认虚拟环境已正确激活，并检查 pip 版本：

```bash
# 检查 pip 版本（建议升级到最新版本）
pip --version

# 升级 pip（可选，但推荐）
pip install --upgrade pip
```

#### 1.5 退出虚拟环境（可选）

当不再需要虚拟环境时，可以退出：

```bash
deactivate
```

**提示**：每次使用项目时，都需要先激活虚拟环境。可以将激活命令添加到 shell 配置文件中（如 `~/.zshrc` 或 `~/.bashrc`）以便快速激活。

---

### 步骤 2：安装项目依赖

在激活虚拟环境后，按照以下步骤安装依赖。

#### 2.1 Python 版本要求

- **Python 3.8+** (推荐 3.10)

#### 2.2 安装基础依赖

```bash
# PyTorch（根据你的系统选择 CPU 或 GPU 版本）
# macOS (MPS):
pip install torch torchvision

# Linux/Windows (CUDA):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 基础科学计算库
pip install numpy pandas scipy

# YAML 配置文件解析
pip install pyyaml
```

#### 2.6 一键安装所有依赖（推荐）

如果你想要一次性安装所有必需的依赖，可以使用以下命令：

```bash
# 确保虚拟环境已激活
# source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 安装所有必需依赖
pip install torch torchvision numpy pandas scipy pyyaml
pip install Flask==2.3.3 flask-socketio==5.3.5 eventlet==0.33.3
pip install psutil matplotlib scikit-learn

# 可选：模型导出依赖
pip install onnx
```

#### 2.7 验证安装

安装完成后，可以验证关键依赖是否正确安装：

```bash
# 验证 PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# 验证 Flask
python -c "import flask; print(f'Flask version: {flask.__version__}')"

# 验证其他关键库
python -c "import numpy, pandas, scipy, yaml; print('All dependencies installed successfully!')"
```

### 重要说明

⚠️ **本系统不再依赖 Fed-BioMed 框架**。所有联邦学习逻辑都在 `federated_simulation_trainer.py` 中自研实现，采用单进程模拟多客户端的方式。


## 快速开始

### 0. 数据集说明

本项目使用 **uci2_dataset** 数据集，包含：
- 3个fold的CSV特征文件（`feat_fold_0.csv`, `feat_fold_1.csv`, `feat_fold_2.csv`）
- 每个文件包含222个特征列
- 数据位于项目根目录的 `uci2_dataset/` 目录下

数据集详细信息请参考 `Waveform-classification/help_files/UCI2_DATASET_ANALYSIS.md`

### 1. 数据准备

首先需要准备数据并分割到多个节点：

```bash
cd federate_waveform
python prepare_federated_data.py
```

这将：
1. 从 `uci2_dataset/` 加载所有fold的数据
2. 创建低血压标签（SP < 90 或 DP < 60）
3. 合并所有fold并分割为训练/验证/测试集
4. 将训练数据分配到多个节点（node_1, node_2, node_3）
5. 创建 `federated_data/` 目录，包含每个节点的数据文件

**输出文件**：
- `federated_data/node_1/train.pth` - 节点1的训练/验证数据分割
- `federated_data/node_2/train.pth` - 节点2的训练/验证数据分割
- `federated_data/node_3/train.pth` - 节点3的训练/验证数据分割
- `federated_data/test.pth` - 测试集数据

**注意**: 确保 `uci2_dataset/` 目录在项目根目录下，且包含 `feat_fold_0.csv`, `feat_fold_1.csv`, `feat_fold_2.csv` 文件。

### 2. 配置设备参数

编辑 `devices.yaml` 文件，配置每个节点的硬件参数和在线模式：

```yaml
devices:
  - id: node_1
    type: "phone_low_end"
    compute_power: "low"
    online_pattern: "intermittent"    # 间歇在线（约40%轮次参与）
    # ... 其他参数

  - id: node_2
    type: "tablet_mid"
    compute_power: "medium"
    online_pattern: "mostly_online"   # 大部分时间在线（约80%轮次参与）
    # ... 其他参数

  - id: node_3
    type: "edge_gateway"
    compute_power: "high"
    online_pattern: "always_on"       # 始终在线（100%参与）
    # ... 其他参数
```

### 3. 启动可视化监控系统

```bash
# 启动可视化监控服务器
python federated_learning_visualization.py
```

访问 `http://localhost:5002` 打开控制面板，可以：
- **激活节点**: 点击"启动节点"按钮激活 node_1, node_2, node_3
- **配置训练**: 设置训练轮数、批次大小、学习率等参数
- **启用压缩**（可选）: 在训练配置中启用模型量化、剪枝等压缩功能
- **启动训练**: 点击"开始训练"按钮启动联邦训练
- **实时监控**: 查看训练进度、指标曲线、节点状态等
- **实时数据**: 训练过程中，数据自动保存到 `results/` 文件夹
- **数据分析**: 查看训练历史和收敛分析

**训练过程中的自动保存**:
- 每轮训练完成后，自动保存到 `results/training_data/round_N.json`
- 每轮模型自动保存到 `results/models/current_round_N.pth`
- 最佳模型自动跟踪并保存到 `results/models/best_model.pth`
- 如果启用压缩，压缩模型保存到 `results/compressed_models/round_N/`

**详细使用说明请参考：[可视化监控系统使用手册](VISUALIZATION_USER_MANUAL.md)**

### 4. 运行实验（命令行方式）

#### 基础实验

```bash
python federated_hypotension_experiment.py
```

#### 带监控的实验

```bash
python federated_hypotension_experiment_with_monitor.py
```

访问 `http://localhost:5000` 查看实时监控界面。

## 功能模块说明

### 1. 自研联邦训练核心 (`federated_simulation_trainer.py`)

这是系统的核心模块，实现了完整的联邦学习训练循环：

- **客户端抽象**: `FederatedClient` 类封装单个客户端的训练逻辑
- **FedAvg 聚合**: `federated_average()` 函数实现模型聚合
- **训练循环**: `run_federated_training()` 函数协调整个训练过程
- **设备仿真**: 根据 `devices.yaml` 配置模拟不同设备的计算能力和在线模式

### 2. 硬件感知优化

系统支持根据设备性能自动调整训练参数：

```python
# devices.yaml 中配置
compute_power: "low"      # low, medium, high
cpu_threads: 1            # PyTorch 线程数
max_batch_size: 32        # 最大批次大小
simulated_compute_delay_ms: 180  # 模拟计算延迟
```

### 3. 设备在线模式

支持四种在线模式，模拟真实场景中的设备可用性：

- `always_on`: 100% 参与率（如边缘网关）
- `mostly_online`: 约 80% 参与率（如平板设备）
- `intermittent`: 约 40% 参与率（如低端手机）
- `sporadic`: 约 20% 参与率（如IoT设备）

训练顺序按在线模式的可靠性优先级排序：`always_on` > `mostly_online` > `intermittent` > `sporadic`

### 4. 模型压缩和导出

系统支持多种模型优化技术，可以在训练过程中自动应用：

#### 4.1 启用模型压缩

在训练配置中启用压缩功能：

```python
from federated_simulation_trainer import run_federated_training

config = {
    'round_limit': 5,
    'training_args': {...},
    'model_args': {...}
}

# 压缩配置
compression_config = {
    'quantization': {
        'enabled': True,
        'type': 'dynamic'  # 或 'static'
    },
    'pruning': {
        'enabled': True,
        'ratio': 0.3,      # 剪枝比例
        'type': 'l1_unstructured'  # 或 'l2_unstructured'
    }
}

result = run_federated_training(
    config=config,
    save_results=True,
    enable_compression=True,
    compression_config=compression_config
)
```

#### 4.2 模型量化

**动态量化**（推荐，无需校准数据）:
```python
from model_quantization import ModelQuantizer

quantizer = ModelQuantizer()
quantized_model = quantizer.quantize_model(
    model=original_model,
    quantization_type='dynamic'
)
```

**静态量化**（需要校准数据，压缩率更高）:
```python
quantized_model = quantizer.quantize_model(
    model=original_model,
    quantization_type='static',
    calibration_data=calibration_loader
)
```

#### 4.3 模型剪枝

```python
from model_pruning import ModelPruner

pruner = ModelPruner()
pruned_model = pruner.prune_model(
    model=original_model,
    pruning_ratio=0.3,  # 剪枝30%的参数
    pruning_type='l1_unstructured'  # 或 'l2_unstructured'
)

# 获取剪枝统计
stats = pruner.get_pruning_statistics(original_model, pruned_model)
print(f"参数减少: {stats['parameter_reduction_percent']:.2f}%")
```

#### 4.4 知识蒸馏

```python
from model_distillation import KnowledgeDistillation

# 创建教师模型（大模型）和学生模型（小模型）
teacher_model = myecgnet(pretrained=False, num_classes=1)
student_model = myecgnet(pretrained=False, num_classes=1)  # 可以是更小的架构

distiller = KnowledgeDistillation(
    teacher_model=teacher_model,
    student_model=student_model,
    temperature=3.0,  # 温度参数
    alpha=0.7         # 蒸馏损失权重
)

# 训练学生模型
for epoch in range(num_epochs):
    for data, labels in train_loader:
        result = distiller.train_step(data, labels, optimizer)
        print(f"Loss: {result['loss']:.4f}, Accuracy: {result['accuracy']:.4f}")
```

#### 4.5 模型导出

```python
from model_export import ModelExporter

exporter = ModelExporter()
exported_files = exporter.export_all_formats(
    model=result['best_model'],  # 使用最佳模型
    input_shape=(1, 1, 1, 1000),
    output_dir='./models',
    model_name='hypotension_model'
)
# 输出: {'onnx': 'models/hypotension_model.onnx', 'torchscript': '...', ...}
```

详细使用方法请参考各模块的文档字符串。

## 配置说明

### 训练配置

在可视化界面或代码中可以配置：

- `rounds`: 联邦训练轮数（默认5）
- `batch_size`: 批次大小（默认128）
- `learning_rate`: 学习率（默认4e-5）
- `epochs`: 每轮训练的本地epoch数（默认1）

### 设备配置 (`devices.yaml`)

每个设备可以配置：

- `online_pattern`: 在线模式（always_on, mostly_online, intermittent, sporadic）
- `compute_power`: 计算能力（low, medium, high）
- `cpu_threads`: CPU线程数
- `max_batch_size`: 最大批次大小
- `simulated_compute_delay_ms`: 模拟计算延迟（毫秒）
- `upload_latency_ms`: 上传延迟（毫秒）
- `bandwidth_kbps`: 带宽（Kbps）

## 📊 可产出的数据/模型（用于报告）

训练完成后，系统可以产出以下数据用于撰写报告：

### 1. 训练历史数据（实时保存）

**位置**: 
- `results/training_data/training_history.json` (实时更新)
- `results/training_data/round_N.json` (每轮单独保存)
- `run_federated_training()` 返回的 `result['history']`

**内容**:
```python
training_history = [
    {
        'round': 0,
        'global_metrics': {
            'loss': 0.5234,
            'val_loss': 0.5123,
            'accuracy': 0.7567,
            'f1': 0.7234
        },
        'client_metrics': {
            'node_1': {'loss': 0.5, 'accuracy': 0.75, 'f1': 0.72, ...},
            'node_2': {'loss': 0.55, 'accuracy': 0.74, 'f1': 0.71, ...},
            'node_3': {'loss': 0.52, 'accuracy': 0.76, 'f1': 0.73, ...}
        },
        'time': 125.3  # 本轮耗时（秒）
    },
    # ... 更多轮次
]
```

**用途**: 
- 绘制训练曲线（Loss、Accuracy、F1 Score）
- 分析收敛速度
- 对比不同节点的性能
- 计算平均训练时间

### 2. 全局指标数据（实时保存）

**位置**: 
- `results/training_data/round_N.json` (每轮实时保存)
- `results/training_data/training_history.json` (实时更新)
- `monitor.state['global_metrics']` 或 `result['history']` 中每轮的 `global_metrics`

**内容**:
```python
global_metrics = {
    'loss': [0.5234, 0.5123, 0.5012, ...],      # 每轮的loss
    'f1': [0.7234, 0.7345, 0.7456, ...],       # 每轮的F1 score
    'accuracy': [0.7567, 0.7654, 0.7743, ...],  # 每轮的accuracy
    'rounds': [0, 1, 2, 3, 4]                    # 轮次索引
}
```

**用途**:
- 绘制全局指标曲线
- 分析模型收敛趋势
- 评估最终模型性能

### 3. 模型文件（自动保存）

**位置**: `results/models/` 目录

**内容**: 
- **每轮模型**: `current_round_N.pth` - 每轮训练完成后的模型
- **最佳模型**: `best_model.pth` - 自动跟踪的最佳模型（基于 F1 Score）
- **最佳模型信息**: `best_model_info.json` - 包含最佳轮次、F1 Score、Accuracy 等
- **最终模型**: `final_model.pth` - 训练结束时的模型

**自动保存机制**:
- 每轮训练完成后，自动保存 `current_round_N.pth`
- 如果当前轮的 F1 Score 超过历史最佳值，自动更新 `best_model.pth`
- 训练结束后，保存 `final_model.pth`

**使用方式**:
```python
# 加载最佳模型
best_model_state = torch.load('results/models/best_model.pth')

# 查看最佳模型信息
import json
with open('results/models/best_model_info.json', 'r') as f:
    best_info = json.load(f)
print(f"Best model at round {best_info['round']}, F1: {best_info['f1_score']:.4f}")

# 导出最佳模型
from model_export import ModelExporter
exporter = ModelExporter()
exporter.export_to_onnx(
    model=best_model_state,  # 需要先加载到模型实例
    input_shape=(1, 1, 1, 1000),
    output_path='best_model.onnx'
)
```

**用途**:
- 模型部署
- 模型性能评估
- 模型压缩实验（量化、剪枝等）

### 4. 最终评估指标（自动保存）

**位置**: 
- `results/training_data/final_metrics.json` (自动保存)
- `run_federated_training()` 返回的 `result['final_metrics']`

**内容**:
```python
final_metrics = {
    'node_1': {
        'loss': 0.5012,
        'accuracy': 0.7743,
        'f1': 0.7456,
        'samples': 109492
    },
    'node_2': {...},
    'node_3': {...}
}
```

**用途**:
- 报告最终模型在各节点上的性能
- 对比不同节点的数据分布影响
- 评估模型泛化能力

### 5. 训练总结（自动保存）

**位置**: `results/training_data/training_summary.json`

**内容**:
```json
{
    "total_rounds": 5,
    "best_round": 3,
    "best_f1_score": 0.7456,
    "final_metrics": {...},
    "training_time": 625.3,
    "timestamp": "2024-12-09 18:30:00"
}
```

### 6. 监控状态数据

**位置**: `monitor.state` 或通过 `monitor.get_state()` 获取

**内容**:
```python
{
    'experiment_running': False,
    'current_round': 4,
    'total_rounds': 5,
    'start_time': '2024-12-09T18:00:00',
    'end_time': '2024-12-09T18:15:30',
    'experiment_config': {...},
    'nodes': {
        'node_1': {
            'status': 'completed',
            'data_size': 109492,
            'metrics': {...}
        },
        # ...
    },
    'round_history': [...],  # 每轮的详细历史
    'global_metrics': {...},  # 全局指标数组
    'detailed_status': {
        'round_start_time': '...',
        'round_times': [125.3, 118.7, ...],  # 每轮耗时
        'current_metrics': {...},
        'nodes_training': {...}
    }
}
```

**保存方式**:
```python
import json
state = monitor.get_state()
with open('training_state.json', 'w') as f:
    json.dump(state, f, indent=2)
```

**用途**:
- 完整的训练过程记录
- 节点参与情况分析
- 训练时间统计
- 可视化数据源

### 7. 压缩模型文件（如果启用压缩）

**位置**: `results/compressed_models/round_N/`

**内容**:
- `quantized_model.pth`: 量化后的模型（如果启用量化）
- `pruned_model.pth`: 剪枝后的模型（如果启用剪枝）

**说明**: 每轮训练完成后，如果启用了压缩功能，会自动生成压缩模型并保存。

### 8. 可视化图表数据

**通过前端界面导出**:
- 训练曲线图（Loss、Accuracy、F1 Score）
- 节点状态图
- 收敛分析图

**或使用代码生成**:
```python
from convergence_monitoring import ConvergenceMonitor

monitor = ConvergenceMonitor()
# 从 training_history 填充数据
for round_data in training_history:
    monitor.record_round(
        round_num=round_data['round'],
        loss=round_data['global_metrics']['loss'],
        f1_score=round_data['global_metrics']['f1'],
        accuracy=round_data['global_metrics']['accuracy']
    )

# 生成图表
monitor.plot_convergence_curves(save_path='convergence.png')

# 生成报告
report = monitor.get_convergence_report()
```

### 9. 模型导出文件

使用 `model_export.py` 模块可以导出：

- **PyTorch 模型** (`.pth`): 完整模型状态
- **ONNX 模型** (`.onnx`): 跨平台推理格式
- **TorchScript 模型** (`.pt`): PyTorch 序列化格式
- **TensorFlow Lite 模型** (`.tflite`): 移动端部署格式

**示例**:
```python
from model_export import ModelExporter

exporter = ModelExporter()
exported_files = exporter.export_all_formats(
    model=result['global_model'],
    input_shape=(1, 1, 1, 1000),
    output_dir='./models',
    model_name='hypotension_model'
)
# 输出: {'onnx': 'models/hypotension_model.onnx', ...}
```

### 10. 数据统计信息

**节点数据分布**:
- 每个节点的样本数量（从 `train.pth` 文件读取）
- 正负样本比例（低血压 vs 正常）
- 数据特征统计

**训练统计**:
- 总训练时间
- 每轮平均时间
- 节点参与率（根据 online_pattern 计算）

## 输出文件结构

训练过程中，系统会自动在 `results/` 文件夹中保存以下数据：

```
results/
├── models/                          # 模型文件目录
│   ├── current_round_0.pth         # 第1轮的当前模型
│   ├── current_round_1.pth         # 第2轮的当前模型
│   ├── current_round_2.pth         # 第3轮的当前模型
│   ├── ...                          # 每轮都会保存 current 模型
│   ├── best_model.pth              # 最佳模型（自动更新）
│   ├── best_model_info.json        # 最佳模型信息（轮次、指标等）
│   └── final_model.pth             # 最终模型（训练结束时的模型）
│
├── training_data/                   # 训练数据目录
│   ├── round_0.json                 # 第1轮的详细数据
│   ├── round_1.json                 # 第2轮的详细数据
│   ├── round_2.json                 # 第3轮的详细数据
│   ├── ...                          # 每轮都会实时保存
│   ├── training_history.json        # 完整训练历史（每轮更新）
│   ├── final_metrics.json           # 最终评估指标
│   └── training_summary.json        # 训练总结（最佳轮次、总时间等）
│
├── compressed_models/               # 压缩模型目录（如果启用压缩）
│   ├── round_0/
│   │   ├── quantized_model.pth     # 量化模型
│   │   └── pruned_model.pth        # 剪枝模型
│   ├── round_1/
│   │   ├── quantized_model.pth
│   │   └── pruned_model.pth
│   └── ...
│
└── figures/                         # 可视化图表（手动生成）
    ├── convergence_curves.png       # 收敛曲线
    ├── loss_curve.png               # Loss 曲线
    ├── accuracy_curve.png          # Accuracy 曲线
    └── f1_curve.png                 # F1 Score 曲线
```

### 实时数据保存说明

- **每轮训练完成后立即保存**:
  - `round_N.json`: 包含该轮的全局指标、客户端指标、训练时间等
  - `current_round_N.pth`: 该轮的模型状态字典
  - `training_history.json`: 自动更新，包含所有轮次的历史

- **最佳模型自动跟踪**:
  - 系统使用 **F1 Score** 作为主要指标来判断最佳模型
  - 当某轮的 F1 Score 超过当前最佳值时，自动保存为 `best_model.pth`
  - 同时更新 `best_model_info.json`，记录最佳轮次和指标

- **训练结束后保存**:
  - `final_model.pth`: 最后一轮的模型
  - `final_metrics.json`: 所有节点的最终评估指标
  - `training_summary.json`: 训练总结，包括最佳轮次、总训练时间等

## 报告撰写建议

基于产出的数据，可以撰写以下内容：

1. **实验设置**
   - 数据集描述（uci2_dataset，样本数，特征数）
   - 设备配置（3个节点，不同计算能力和在线模式）
   - 训练参数（轮数、批次大小、学习率）
   - 模型压缩配置（如果启用）

2. **实验结果**
   - 训练曲线图（使用 `results/training_data/training_history.json`）
   - 最终性能指标（使用 `results/training_data/final_metrics.json`）
   - 最佳模型性能（使用 `results/models/best_model_info.json`）
   - 收敛分析（使用每轮的 `round_N.json` 数据）

3. **性能分析**
   - 不同节点的性能对比（从 `round_N.json` 中的 `client_metrics` 提取）
   - 在线模式对训练的影响（对比不同 online_pattern）
   - 训练时间统计（从 `training_summary.json` 获取）
   - 最佳模型出现时机（从 `best_model_info.json` 获取）

4. **模型分析**
   - 模型大小对比（原始模型 vs 压缩模型）
   - 模型压缩效果（量化、剪枝前后对比，从 `compressed_models/` 目录获取）
   - 最佳模型 vs 最终模型性能对比
   - 模型导出格式兼容性

5. **实时数据优势**
   - 说明系统支持实时数据保存，每轮训练完成后立即保存
   - 展示如何从 `results/` 目录中提取数据进行分析
   - 说明最佳模型自动跟踪机制的优势

## 故障排除

### 常见问题

1. **MPS Tensor float64 错误**
   - 原因: macOS MPS 不支持 float64
   - 解决: 代码已自动处理，确保所有张量使用 float32

2. **数据路径错误**
   - 确保 `uci2_dataset/` 目录在项目根目录下
   - 确保包含 `feat_fold_0.csv`, `feat_fold_1.csv`, `feat_fold_2.csv` 文件
   - 检查 `federated_data/` 目录下是否有各节点的 `train.pth` 文件

3. **端口被占用**
   - 修改 `federated_learning_visualization.py` 中的端口号（默认 5002）

4. **节点未参与训练**
   - 检查节点是否已激活（在可视化界面中点击"启动节点"）
   - 检查 `devices.yaml` 中的 `online_pattern` 配置
   - 查看日志中的节点可用性信息

## 性能优化建议

1. **硬件感知**: 使用 `devices.yaml` 配置不同设备的计算能力，系统会自动调整批次大小
2. **资源调度**: 训练顺序按在线模式优先级自动排序，确保可靠设备优先参与
3. **模型压缩**: 使用量化、剪枝等技术减少模型大小和推理时间
4. **自适应调整**: 根据网络条件和设备状态动态调整训练策略

## 数据集信息

### uci2_dataset

- **数据来源**: UCI2数据集
- **数据格式**: CSV特征文件（222个特征）
- **数据规模**: 约41万样本，分布在3个fold中
- **特征类型**: PPG、VPG、APG信号特征，时间域特征，面积特征等
- **标签**: 低血压标签（基于SP < 90 或 DP < 60）

详细分析请参考: `Waveform-classification/help_files/UCI2_DATASET_ANALYSIS.md`

### 数据准备流程

1. `prepare_federated_data.py` 加载所有fold的CSV文件
2. 创建patient_trial格式的ID
3. 根据SP和DP创建低血压标签
4. 合并所有fold并随机分割为训练/验证/测试集
5. 将训练数据分配到多个节点（node_1, node_2, node_3）

## 系统架构说明

本系统采用**自研联邦学习框架**，不再依赖 Fed-BioMed：

- **单进程模拟**: 在一个 Python 进程中模拟多个客户端
- **逻辑抽象**: 节点（node）和研究者（researcher）都是逻辑抽象，不是独立进程
- **直接调用**: 训练逻辑通过函数调用直接执行，无需网络通信
- **实时监控**: 通过 Flask + SocketIO 实现实时状态推送

详细架构说明请参考: `BIG_CHANGE.md`

## 参考文献

- McMahan, B., et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- 差分隐私: Dwork, C. (2006). "Differential Privacy"
- UCI2数据集: 用于低血压预测的医疗波形数据集

## 许可证

本项目遵循相应的开源许可证。

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

---

**最后更新**: 2025年12月  
**版本**: v2.0 (自研联邦学习框架版本)
