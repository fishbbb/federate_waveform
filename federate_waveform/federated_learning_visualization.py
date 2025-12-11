# -*- coding: utf-8 -*-
"""
联邦学习可视化监控模块
用于监控和展示联邦学习训练过程
"""

import os
import json
import time
import threading
import subprocess
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict
from pathlib import Path

# 导入torch用于读取.pth文件
try:
    import torch
except ImportError:
    torch = None
    print("警告: torch未安装，无法读取数据文件信息")

# 导入torch用于读取.pth文件
try:
    import torch
except ImportError:
    torch = None
    print("警告: torch未安装，无法读取数据文件信息")

# 检查并导入Flask相关依赖
try:
    from flask import Flask, render_template, jsonify, request
except ImportError:
    print("=" * 60)
    print("错误: Flask未安装")
    print("=" * 60)
    print("请运行: pip install Flask==2.3.3")
    print("或运行: pip install -r visualization_requirements.txt")
    print("=" * 60)
    raise

try:
    from flask_socketio import SocketIO, emit
except ImportError:
    print("=" * 60)
    print("错误: flask-socketio未安装")
    print("=" * 60)
    print("请运行: pip install flask-socketio==5.3.5")
    print("或运行: pip install -r visualization_requirements.txt")
    print("=" * 60)
    raise

# 尝试使用eventlet，如果不可用则使用threading
try:
    import eventlet
    # 使用eventlet作为异步后端
    eventlet.monkey_patch()
    async_mode = 'eventlet'
except ImportError:
    async_mode = 'threading'
    print("警告: eventlet未安装，使用threading模式。")
    print("为获得更好性能，建议安装: pip install eventlet==0.33.3")

# 获取当前文件所在目录
# 从federate_waveform文件夹向上到根目录，然后访问visualization
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)  # 向上到根目录
template_dir = os.path.join(base_dir, 'visualization', 'templates')
static_dir = os.path.join(base_dir, 'visualization', 'static')

# 验证路径
if not os.path.exists(template_dir):
    print(f"警告: 模板目录不存在: {template_dir}")
    print(f"当前目录: {current_dir}")
    print(f"基础目录: {base_dir}")
if not os.path.exists(static_dir):
    print(f"警告: 静态文件目录不存在: {static_dir}")

# 确保目录存在
if not os.path.exists(template_dir):
    raise FileNotFoundError(f"Template directory not found: {template_dir}")
if not os.path.exists(static_dir):
    raise FileNotFoundError(f"Static directory not found: {static_dir}")

app = Flask(__name__, 
            template_folder=template_dir,
            static_folder=static_dir)
app.config['SECRET_KEY'] = 'federated-learning-visualization'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)

# 全局状态存储
training_state = {
    'experiment_running': False,
    'current_round': 0,
    'total_rounds': 0,
    'nodes': {},  # {node_id: {status, data_size, metrics, ...}}
    'round_history': [],  # [{round, timestamp, nodes_metrics, global_metrics}]
    'global_metrics': {
        'loss': [],
        'f1': [],
        'accuracy': [],
        'rounds': []
    },
    'detailed_status': {
        'round_start_time': None,  # 当前轮次开始时间
        'round_times': [],  # 每轮耗时（秒）
        'current_metrics': {
            'loss': None,
            'f1': None,
            'accuracy': None
        },
        'nodes_training': {}  # {node_id: {status, start_time, progress}}
    },
    'start_time': None,
    'end_time': None,
    'experiment_config': {},
    'active_nodes': set(),  # 激活的节点集合（不再使用外部进程）
    'logs': [],  # 日志列表
    'analysis_data': {}  # 分析数据
}

# 节点状态枚举
NODE_STATUS = {
    'IDLE': 'idle',
    'TRAINING': 'training',
    'UPLOADING': 'uploading',
    'COMPLETED': 'completed',
    'ERROR': 'error'
}


class FederatedLearningMonitor:
    """联邦学习监控器"""
    
    def __init__(self):
        self.state = training_state
        self.callbacks = []
    
    def register_callback(self, callback):
        """注册状态更新回调"""
        self.callbacks.append(callback)
    
    def notify_callbacks(self, event_type, data):
        """通知所有回调函数"""
        for callback in self.callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                print(f"Error in callback: {e}")
    
    def start_experiment(self, config: Dict):
        """开始实验"""
        self.state['experiment_running'] = True
        self.state['current_round'] = 0
        self.state['total_rounds'] = config.get('round_limit', 5)
        self.state['start_time'] = datetime.now().isoformat()
        self.state['experiment_config'] = config
        self.state['nodes'] = {}
        self.state['round_history'] = []
        
        # 初始化全局指标数组（预分配空间）
        total_rounds = config.get('round_limit', 5)
        self.state['global_metrics'] = {
            'loss': [0.0] * total_rounds,
            'f1': [0.0] * total_rounds,
            'accuracy': [0.0] * total_rounds,
            'rounds': list(range(total_rounds))
        }
        
        # 初始化详细状态
        self.state['detailed_status'] = {
            'round_start_time': None,
            'round_times': [],
            'current_metrics': {
                'loss': None,
                'f1': None,
                'accuracy': None
            },
            'nodes_training': {}
        }
        
        print(f"实验已启动: {total_rounds} 轮训练")
        print(f"指标数组已初始化: loss={len(self.state['global_metrics']['loss'])}, f1={len(self.state['global_metrics']['f1'])}, accuracy={len(self.state['global_metrics']['accuracy'])}")
        
        self.notify_callbacks('experiment_started', {
            'config': config,
            'timestamp': self.state['start_time']
        })
    
    def end_experiment(self):
        """结束实验"""
        self.state['experiment_running'] = False
        self.state['end_time'] = datetime.now().isoformat()
        
        self.notify_callbacks('experiment_ended', {
            'timestamp': self.state['end_time']
        })
    
    def update_node_status(self, node_id: str, status: str, **kwargs):
        """更新节点状态"""
        # 验证节点ID是否有效
        if node_id not in ['node_1', 'node_2', 'node_3']:
            # 尝试标准化节点ID
            if node_id.startswith('node_'):
                # 已经是标准格式，但不在有效列表中，可能是无效ID
                return
            # 尝试提取数字
            import re
            num_match = re.search(r'(\d+)', node_id)
            if num_match and int(num_match.group(1)) in [1, 2, 3]:
                node_id = f'node_{num_match.group(1)}'
            else:
                # 无效的节点ID，忽略
                return
        
        if node_id not in self.state['nodes']:
            self.state['nodes'][node_id] = {
                'id': node_id,
                'status': status,
                'data_size': 0,
                'metrics': {},
                'last_update': datetime.now().isoformat()
            }
        
        self.state['nodes'][node_id]['status'] = status
        self.state['nodes'][node_id]['last_update'] = datetime.now().isoformat()
        
        for key, value in kwargs.items():
            self.state['nodes'][node_id][key] = value
        
        self.notify_callbacks('node_status_updated', {
            'node_id': node_id,
            'status': status,
            'data': self.state['nodes'][node_id]
        })
    
    def start_round(self, round_num: int):
        """开始新的一轮训练"""
        # 记录上一轮的时间（如果有）
        if 'detailed_status' in self.state and self.state['detailed_status'].get('round_start_time'):
            prev_round_time = (datetime.now() - datetime.fromisoformat(
                self.state['detailed_status']['round_start_time']
            )).total_seconds()
            if 'round_times' not in self.state['detailed_status']:
                self.state['detailed_status']['round_times'] = []
            self.state['detailed_status']['round_times'].append(prev_round_time)
        
        # 记录当前轮次开始时间
        if 'detailed_status' not in self.state:
            self.state['detailed_status'] = {
                'round_start_time': None,
                'round_times': [],
                'current_metrics': {'loss': None, 'f1': None, 'accuracy': None},
                'nodes_training': {}
            }
        self.state['detailed_status']['round_start_time'] = datetime.now().isoformat()
        
        self.state['current_round'] = round_num
        
        # 确保轮次指标数组足够大
        while len(self.state['global_metrics']['rounds']) <= round_num:
            self.state['global_metrics']['rounds'].append(len(self.state['global_metrics']['rounds']))
            self.state['global_metrics']['loss'].append(0.0)
            self.state['global_metrics']['f1'].append(0.0)
            self.state['global_metrics']['accuracy'].append(0.0)
        
        # 重置所有节点状态为 idle（训练开始后会更新为 training）
        for node_id in self.state['nodes']:
            self.state['nodes'][node_id]['status'] = NODE_STATUS['IDLE']
        
        self.notify_callbacks('round_started', {
            'round': round_num,
            'timestamp': datetime.now().isoformat()
        })
    
    def update_round_metrics(self, round_num: int, node_metrics: Dict, global_metrics: Optional[Dict] = None):
        """更新轮次指标"""
        round_data = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'nodes': node_metrics,
            'global': global_metrics or {}
        }
        
        # 添加到历史记录
        while len(self.state['round_history']) <= round_num:
            self.state['round_history'].append({
                'round': len(self.state['round_history']),
                'timestamp': datetime.now().isoformat(),
                'nodes': {},
                'global': {}
            })
        self.state['round_history'][round_num] = round_data
        
        # 更新全局指标（确保数组长度一致）
        while len(self.state['global_metrics']['rounds']) <= round_num:
            self.state['global_metrics']['rounds'].append(len(self.state['global_metrics']['rounds']))
            self.state['global_metrics']['loss'].append(0.0)
            self.state['global_metrics']['f1'].append(0.0)
            self.state['global_metrics']['accuracy'].append(0.0)
        
        # 更新指定轮次的指标
        if global_metrics:
            if 'loss' in global_metrics:
                self.state['global_metrics']['loss'][round_num] = global_metrics['loss']
            if 'f1' in global_metrics:
                self.state['global_metrics']['f1'][round_num] = global_metrics['f1']
            if 'accuracy' in global_metrics:
                self.state['global_metrics']['accuracy'][round_num] = global_metrics['accuracy']
        
        # 更新详细状态中的当前轮次指标（这是前端显示的地方）
        if 'detailed_status' not in self.state:
            self.state['detailed_status'] = {
                'round_start_time': None,
                'round_times': [],
                'current_metrics': {'loss': None, 'f1': None, 'accuracy': None},
                'nodes_training': {}
            }
        
        if global_metrics:
            self.state['detailed_status']['current_metrics'] = {
                'loss': global_metrics.get('loss'),
                'f1': global_metrics.get('f1'),
                'accuracy': global_metrics.get('accuracy')
            }
        
        # 更新节点训练状态
        if 'nodes_training' not in self.state['detailed_status']:
            self.state['detailed_status']['nodes_training'] = {}
        
        for node_id, metrics in node_metrics.items():
            self.state['detailed_status']['nodes_training'][node_id] = {
                'status': 'completed',
                'loss': metrics.get('loss'),
                'f1': metrics.get('f1'),
                'accuracy': metrics.get('accuracy'),
                'samples': metrics.get('samples', 0)
            }
        
        self.notify_callbacks('round_metrics_updated', round_data)
    
    def get_state(self) -> Dict:
        """获取当前状态（排除不可序列化的对象）"""
        state = self.state.copy()
        
        # 移除不可序列化的对象
        # training_process 和 node_processes 包含 Popen 对象，不能序列化
        # 只返回进程的 PID 信息
        # 确保 current_round 和 total_rounds 有默认值
        current_round = state.get('current_round', 0)
        total_rounds = state.get('total_rounds', 0)
        
        # 如果 total_rounds 为 0 但 experiment_config 存在，尝试从配置中获取
        if total_rounds == 0 and state.get('experiment_config'):
            total_rounds = state.get('experiment_config', {}).get('round_limit', 0)
        
        serializable_state = {
            'experiment_running': state.get('experiment_running', False),
            'current_round': current_round,
            'total_rounds': total_rounds,
            'start_time': state.get('start_time'),
            'end_time': state.get('end_time'),
            'experiment_config': state.get('experiment_config', {}),
            'nodes': state.get('nodes', {}),
            'round_history': state.get('round_history', []),
            'global_metrics': state.get('global_metrics', {
                'loss': [],
                'f1': [],
                'accuracy': [],
                'rounds': []
            }),
            'detailed_status': state.get('detailed_status', {
                'round_start_time': None,
                'round_times': [],
                'current_metrics': {
                    'loss': None,
                    'f1': None,
                    'accuracy': None
                },
                'nodes_training': {}
            })
        }
        
        # 添加进程信息（仅 PID）
        if 'training_process' in state and state['training_process']:
            process = state['training_process']
            serializable_state['training_process_pid'] = process.pid if hasattr(process, 'pid') else None
        else:
            serializable_state['training_process_pid'] = None
        
        # 添加节点进程信息（仅 PID）
        node_processes_info = {}
        if 'node_processes' in state:
            for node_id, process in state['node_processes'].items():
                if process and hasattr(process, 'pid'):
                    node_processes_info[node_id] = {
                        'pid': process.pid,
                        'running': process.poll() is None
                    }
        serializable_state['node_processes'] = node_processes_info
        
        return serializable_state


# 创建全局监控器实例
monitor = FederatedLearningMonitor()


# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    print('Client connected')
    emit('connected', {'message': 'Connected to federated learning monitor'})
    # 发送当前状态
    emit('state_update', monitor.get_state())


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开连接"""
    print('Client disconnected')


@socketio.on('request_state')
def handle_request_state():
    """客户端请求当前状态"""
    emit('state_update', monitor.get_state())


# 监控器回调函数 - 通过WebSocket推送更新
def broadcast_update(event_type, data):
    """广播更新到所有连接的客户端"""
    socketio.emit('update', {
        'event_type': event_type,
        'data': data,
        'timestamp': datetime.now().isoformat()
    })


# 注册回调
monitor.register_callback(broadcast_update)


# HTTP路由
@app.route('/')
def index():
    """主页面"""
    try:
        # 调试信息
        print(f"Template directory: {template_dir}")
        print(f"Template exists: {os.path.exists(template_dir)}")
        index_path = os.path.join(template_dir, 'index.html')
        print(f"Index.html path: {index_path}")
        print(f"Index.html exists: {os.path.exists(index_path)}")
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"Template file size: {len(content)} bytes")
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {e}")
        import traceback
        traceback.print_exc()
        return f"<h1>Error</h1><p>{str(e)}</p><pre>{traceback.format_exc()}</pre>", 500


@app.route('/test')
def test():
    """测试路由"""
    return jsonify({
        'status': 'ok',
        'template_dir': template_dir,
        'static_dir': static_dir,
        'template_exists': os.path.exists(template_dir),
        'static_exists': os.path.exists(static_dir)
    })


@app.route('/simple')
def simple():
    """简单测试页面"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <h1>Server is working!</h1>
        <p>If you see this, the Flask server is running correctly.</p>
        <p><a href="/">Go to main page</a></p>
    </body>
    </html>
    """


@app.route('/api/state')
def get_state():
    """获取当前状态API"""
    return jsonify(monitor.get_state())


@app.route('/api/nodes')
def get_nodes():
    """获取节点信息API"""
    return jsonify({
        'nodes': list(monitor.state['nodes'].values()),
        'count': len(monitor.state['nodes'])
    })


@app.route('/api/metrics')
def get_metrics():
    """获取指标数据API"""
    return jsonify({
        'global_metrics': monitor.state['global_metrics'],
        'round_history': monitor.state['round_history']
    })


@app.route('/api/rounds/<int:round_num>')
def get_round(round_num):
    """获取特定轮次的数据"""
    if round_num < len(monitor.state['round_history']):
        return jsonify(monitor.state['round_history'][round_num])
    return jsonify({'error': 'Round not found'}), 404


# ==================== 控制功能API ====================

# @app.route('/api/nodes/start', methods=['POST'])
# def start_node():
#     """启动节点"""
#     data = request.get_json()
#     node_id = data.get('node_id', 'node_1')
#     node_path = data.get('node_path', f'fbm-node-{node_id.split("_")[-1]}')
    
#     try:
#         base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         node_data_dir = os.path.join(base_dir, 'federated_data', node_id)
#         node_full_path = os.path.join(base_dir, node_path)
        
#         # 检查并确定 fedbiomed 命令路径（优先使用 fb_env 中的）
#         fb_env_path = os.path.join(base_dir, 'fb_env', 'bin', 'fedbiomed')
        
#         # 优先使用 fb_env 中的 fedbiomed
#         if os.path.exists(fb_env_path):
#             fedbiomed_cmd = fb_env_path
#         else:
#             # 回退到系统路径
#             fedbiomed_cmd = 'fedbiomed'
        
#         # 检查节点是否已经在运行
#         if node_id in training_state['node_processes']:
#             process = training_state['node_processes'][node_id]
#             if process.poll() is None:  # 进程还在运行
#                 return jsonify({
#                     'success': False,
#                     'error': f'节点 {node_id} 已经在运行中 (PID: {process.pid})'
#                 }), 400
        
#         # 检查数据目录是否存在
#         if not os.path.exists(node_data_dir):
#             return jsonify({
#                 'success': False,
#                 'error': f'节点数据目录不存在: {node_data_dir}。请先运行 prepare_federated_data.py 准备数据。'
#             }), 404
        
#         # 检查 fedbiomed 命令是否可用（使用已确定的 fedbiomed_cmd）
#         try:
#             subprocess.run([fedbiomed_cmd, '--version'], 
#                          capture_output=True, check=True, timeout=5)
#             add_log(f'✅ fedbiomed 命令可用: {fedbiomed_cmd}', level='info')
#         except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
#             return jsonify({
#                 'success': False,
#                 'error': f'fedbiomed 命令不可用 ({fedbiomed_cmd})。请确保已激活 fb_env 环境并安装了 Fed-BioMed。'
#             }), 500
        
#         # 确保节点目录存在
#         os.makedirs(node_full_path, exist_ok=True)
        
#         # 检查数据集是否已配置
#         dataset_configured = False
#         dataset_count = 0
#         try:
#             import glob
#             dataset_db_path = os.path.join(node_full_path, 'var', 'db_*.json')
#             existing_datasets = glob.glob(dataset_db_path)
#             dataset_count = len(existing_datasets)
#             dataset_configured = dataset_count > 0
#         except Exception as e:
#             add_log(f'检查数据集配置时出错: {e}', level='warning')
        
#         if not dataset_configured:
#             add_log(f'⚠️ 警告: 节点 {node_id} 尚未配置数据集！', level='error')
#             add_log(f'   节点可以启动，但无法参与训练。', level='warning')
#             add_log(f'   配置方法: 在终端运行以下命令（确保在 fb_env 环境中）:', level='info')
#             add_log(f'   source fb_env/bin/activate', level='info')
#             add_log(f'   fedbiomed node --path {node_path} dataset add', level='info')
#             add_log(f'   然后选择数据文件: {node_data_dir}/train.pth', level='info')
#             add_log(f'   标签使用: #hypotension #waveform #ecg #uci2', level='info')
#         else:
#             add_log(f'✅ 节点 {node_id} 已配置 {dataset_count} 个数据集', level='info')
        
#         # 设置环境变量（用于设备仿真）
#         env = os.environ.copy()
#         env['FB_NODE_PATH'] = node_path
        
#         # 启动节点进程
#         # 注意：fedbiomed node start 会持续运行，需要作为后台进程
#         # 使用确定的 fedbiomed 命令路径（优先 fb_env）
#         if sys.platform == 'win32':
#             # Windows平台
#             cmd = [fedbiomed_cmd, 'node', '--path', node_full_path, 'start']
#             process = subprocess.Popen(
#                 cmd,
#                 cwd=node_data_dir,
#                 env=env,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True,
#                 creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
#             )
#         else:
#             # Unix平台（macOS/Linux）- 使用shell以便正确处理路径和环境变量
#             # 使用绝对路径确保使用正确的 fedbiomed
#             cmd = f'cd "{node_data_dir}" && "{fedbiomed_cmd}" node --path "{node_full_path}" start'
#             process = subprocess.Popen(
#                 cmd,
#                 shell=True,
#                 cwd=node_data_dir,
#                 env=env,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True,
#                 preexec_fn=os.setsid if hasattr(os, 'setsid') else None
#             )
        
#         # 等待一小段时间检查进程是否成功启动
#         time.sleep(2)
#         if process.poll() is not None:
#             # 进程已经退出，读取错误信息
#             try:
#                 stdout, stderr = process.communicate(timeout=1)
#             except:
#                 stdout, stderr = '', ''
#             error_msg = (stderr.strip() or stdout.strip() or '进程启动后立即退出')
            
#             # 提供更友好的错误提示
#             if 'fedbiomed' in error_msg.lower() or 'command not found' in error_msg.lower():
#                 error_msg = 'fedbiomed 命令不可用。请确保已激活 fb_env 环境。'
#             elif 'dataset' in error_msg.lower() or 'not found' in error_msg.lower():
#                 error_msg = '数据集未配置。请先运行: fedbiomed node --path ' + node_path + ' dataset add'
            
#             return jsonify({
#                 'success': False,
#                 'error': f'节点启动失败: {error_msg}'
#             }), 500
        
#         training_state['node_processes'][node_id] = process
        
#         # 读取节点实际数据量
#         data_size = 0
#         try:
#             train_pth_path = os.path.join(node_data_dir, 'train.pth')
#             if os.path.exists(train_pth_path):
#                 train_data = torch.load(train_pth_path, map_location='cpu', weights_only=False)
#                 if isinstance(train_data, dict) and 'train' in train_data:
#                     data_size = len(train_data['train'])
#                     add_log(f'节点 {node_id} 数据量: {data_size} 个样本', level='info')
#                 else:
#                     add_log(f'警告: 无法解析节点 {node_id} 的数据文件格式', level='warning')
#             else:
#                 add_log(f'警告: 节点 {node_id} 的数据文件不存在: {train_pth_path}', level='warning')
#         except Exception as e:
#             add_log(f'读取节点 {node_id} 数据量时出错: {str(e)}', level='warning')
#             # 尝试从data_info.txt读取
#             try:
#                 data_info_path = os.path.join(node_data_dir, 'data_info.txt')
#                 if os.path.exists(data_info_path):
#                     with open(data_info_path, 'r') as f:
#                         for line in f:
#                             if 'train' in line.lower() and ('samples' in line.lower() or '样本' in line):
#                                 import re
#                                 numbers = re.findall(r'\d+', line)
#                                 if numbers:
#                                     data_size = int(numbers[0])
#                                     break
#             except:
#                 pass
        
#         # 更新节点状态（使用实际数据量）
#         # 节点刚启动时状态为 'running'，等待连接到 researcher
#         monitor.update_node_status(node_id, 'running', data_size=data_size)
        
#         add_log(f'✅ 节点 {node_id} 已启动 (PID: {process.pid}, Path: {node_path}, 数据量: {data_size})', level='info')
#         add_log(f'   节点正在尝试连接到 researcher...', level='info')
        
#         return jsonify({
#             'success': True,
#             'node_id': node_id,
#             'pid': process.pid,
#             'node_path': node_path,
#             'message': f'节点 {node_id} 启动成功'
#         })
#     except Exception as e:
#         import traceback
#         error_detail = traceback.format_exc()
#         add_log(f'启动节点 {node_id} 失败: {str(e)}\n{error_detail}', level='error')
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500
@app.route('/api/nodes/start', methods=['POST'])
def start_node():
    """启动节点（逻辑抽象，不再启动外部进程）"""
    data = request.get_json()
    node_id = data.get('node_id', 'node_1')
    
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        node_data_dir = os.path.join(base_dir, 'federated_data', node_id)
        
        # 检查节点是否已经激活
        if node_id in training_state.get('active_nodes', set()):
            return jsonify({
                'success': False,
                'error': f'节点 {node_id} 已经激活'
            }), 400
        
        # 检查数据目录是否存在
        if not os.path.exists(node_data_dir):
            return jsonify({
                'success': False,
                'error': f'节点数据目录不存在: {node_data_dir}。请先运行 prepare_federated_data.py 准备数据。'
            }), 404
        
        # 检查数据文件是否存在
        train_pth_path = os.path.join(node_data_dir, 'train.pth')
        if not os.path.exists(train_pth_path):
            return jsonify({
                'success': False,
                'error': f'节点数据文件不存在: {train_pth_path}'
            }), 404
        
        # 读取节点实际数据量
        data_size = 0
        try:
            train_data = torch.load(train_pth_path, map_location='cpu', weights_only=False)
            if isinstance(train_data, dict) and 'train' in train_data:
                data_size = len(train_data['train'])
                add_log(f'节点 {node_id} 数据量: {data_size} 个样本', level='info')
            else:
                add_log(f'警告: 无法解析节点 {node_id} 的数据文件格式', level='warning')
        except Exception as e:
            add_log(f'读取节点 {node_id} 数据量时出错: {str(e)}', level='warning')
            # 尝试从 data_info.txt 读取
            try:
                data_info_path = os.path.join(node_data_dir, 'data_info.txt')
                if os.path.exists(data_info_path):
                    with open(data_info_path, 'r') as f:
                        for line in f:
                            if 'train' in line.lower() and ('samples' in line.lower() or '样本' in line):
                                import re
                                numbers = re.findall(r'\d+', line)
                                if numbers:
                                    data_size = int(numbers[0])
                                    break
            except Exception:
                pass
        
        # 标记节点为激活状态
        if 'active_nodes' not in training_state:
            training_state['active_nodes'] = set()
        training_state['active_nodes'].add(node_id)
        
        # 更新节点状态
        monitor.update_node_status(node_id, 'running', data_size=data_size)
        
        add_log(f'✅ 节点 {node_id} 已激活 (数据量: {data_size})', level='info')
        add_log(f'   节点已准备好参与联邦训练', level='info')
        
        return jsonify({
            'success': True,
            'node_id': node_id,
            'data_size': data_size,
            'message': f'节点 {node_id} 激活成功'
        })
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        add_log(f'激活节点 {node_id} 失败: {str(e)}\n{error_detail}', level='error')
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/nodes/stop', methods=['POST'])
def stop_node():
    """停止节点（取消激活）"""
    data = request.get_json()
    node_id = data.get('node_id')
    
    if not node_id:
        return jsonify({'success': False, 'error': 'node_id is required'}), 400
    
    try:
        if 'active_nodes' in training_state and node_id in training_state['active_nodes']:
            training_state['active_nodes'].remove(node_id)
            monitor.update_node_status(node_id, 'idle')
            add_log(f'节点 {node_id} 已取消激活', level='info')
            
            return jsonify({
                'success': True,
                'message': f'节点 {node_id} 已取消激活'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'节点 {node_id} 未找到或未激活'
            }), 404
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        add_log(f'取消激活节点 {node_id} 失败: {str(e)}\n{error_detail}', level='error')
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/nodes/list', methods=['GET'])
def list_nodes():
    """列出所有节点状态（包括已配置但未启动的节点）"""
    nodes_info = []
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 从devices.yaml读取所有配置的节点
    try:
        devices_yaml = os.path.join(base_dir, 'federate_waveform', 'devices.yaml')
        if os.path.exists(devices_yaml):
            import yaml
            with open(devices_yaml, 'r') as f:
                devices_config = yaml.safe_load(f)
                configured_nodes = {d['id']: d for d in devices_config.get('devices', [])}
        else:
            configured_nodes = {}
    except:
        configured_nodes = {}
    
    # 获取所有可能的节点ID（从配置或默认）
    active_nodes = training_state.get('active_nodes', set())
    all_node_ids = set(configured_nodes.keys()) | active_nodes
    if not all_node_ids:
        # 如果没有配置，使用默认的3个节点
        all_node_ids = {'node_1', 'node_2', 'node_3'}
    
    for node_id in sorted(all_node_ids):
        # 检查激活状态
        if node_id in active_nodes:
            status = 'running'
            pid = None  # 不再使用进程ID
        else:
            status = 'idle'
            pid = None
        
        # 获取节点信息
        node_info = monitor.state['nodes'].get(node_id, {})
        device_config = configured_nodes.get(node_id, {})
        
        # 如果数据量为0，尝试从文件读取
        data_size = node_info.get('data_size', 0)
        if data_size == 0:
            node_data_dir = os.path.join(base_dir, 'federated_data', node_id)
            train_pth_path = os.path.join(node_data_dir, 'train.pth')
            if os.path.exists(train_pth_path) and torch is not None:
                try:
                    train_data = torch.load(train_pth_path, map_location='cpu', weights_only=False)
                    if isinstance(train_data, dict) and 'train' in train_data:
                        data_size = len(train_data['train'])
                        # 更新节点状态
                        monitor.update_node_status(node_id, node_info.get('status', 'idle'), data_size=data_size)
                except Exception as e:
                    # 如果读取失败，尝试从data_info.txt读取
                    try:
                        data_info_path = os.path.join(node_data_dir, 'data_info.txt')
                        if os.path.exists(data_info_path):
                            with open(data_info_path, 'r') as f:
                                import re
                                for line in f:
                                    if 'train' in line.lower() and ('samples' in line.lower() or '样本' in line):
                                        numbers = re.findall(r'\d+', line)
                                        if numbers:
                                            data_size = int(numbers[0])
                                            monitor.update_node_status(node_id, node_info.get('status', 'idle'), data_size=data_size)
                                            break
                    except:
                        pass
        
        nodes_info.append({
            'node_id': node_id,
            'status': status,
            'pid': pid,
            'metrics': node_info.get('metrics', {}),
            'data_size': data_size,
            'device_type': device_config.get('type', 'unknown'),
            'compute_power': device_config.get('compute_power', 'unknown'),
            'online_pattern': device_config.get('online_pattern', 'always_on')
        })
    
    # 过滤掉无效的节点ID（如 node_s）
    valid_nodes = [n for n in nodes_info if n['node_id'] in ['node_1', 'node_2', 'node_3']]
    
    return jsonify({'nodes': valid_nodes})


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """启动训练"""
    data = request.get_json() or {}
    
    try:
        if training_state['experiment_running']:
            return jsonify({'error': 'Training already running'}), 400
        
        # 获取训练参数
        rounds = data.get('rounds', 5)
        batch_size = data.get('batch_size', 128)
        learning_rate = data.get('learning_rate', 4e-5)
        
        # 初始化实验配置
        # 注意：model_args 中的 data_path 仅作为默认值，实际训练时会使用节点上的数据
        config = {
            'round_limit': rounds,
            'tags': ['#hypotension', '#waveform', '#ecg', '#uci2'],
            'training_args': {
                'loader_args': {'batch_size': batch_size},
                'optimizer_args': {'lr': learning_rate},
                'epochs': 1
            },
            'model_args': {
                'batch_size': batch_size,
                'data_path': 'federated_data/node_1/train.pth',  # 仅作为默认值，实际使用节点数据
                'use_uci2': True,
                'uci2_base_dir': 'uci2_dataset'
            }
        }
        
        # 开始监控实验
        monitor.start_experiment(config)
        training_state['experiment_running'] = True
        
        # 确保状态正确初始化
        monitor.state['current_round'] = 0
        monitor.state['total_rounds'] = rounds
        monitor.state['experiment_config'] = config
        
        # 指标数组在start_experiment中已经预分配，这里不需要再次初始化
        
        # 发送初始状态更新（立即发送，确保前端收到）
        socketio.emit('update', {
            'event_type': 'experiment_started',
            'data': {
                'config': config,
                'rounds': rounds,
                'current_round': 0,
                'total_rounds': rounds
            },
            'timestamp': datetime.now().isoformat()
        })
        
        # 立即发送完整状态更新
        state = monitor.get_state()
        socketio.emit('state_update', state)
        
        add_log(f'实验已启动: 总轮数={rounds}, 批次大小={batch_size}, 学习率={learning_rate}', level='info')
        
        # 检查激活的节点
        active_nodes = training_state.get('active_nodes', set())
        if not active_nodes:
            add_log('⚠️ 警告: 没有激活的节点！请先激活节点。', level='error')
            return jsonify({
                'success': False,
                'error': 'No active nodes. Please start nodes first.'
            }), 400
        
        add_log(f'✅ {len(active_nodes)} 个节点已激活: {", ".join(sorted(active_nodes))}', level='info')
        
        # 启动训练（在后台线程中运行）
        def run_training():
            try:
                # 导入自研的联邦训练模块
                from federated_simulation_trainer import run_federated_training
                
                add_log('=' * 60, level='info')
                add_log('🚀 开始联邦训练...', level='info')
                add_log('=' * 60, level='info')
                
                # 直接调用训练函数
                # 传递激活的节点列表（训练顺序将按 online_pattern 优先级决定）
                active_nodes_set = training_state.get('active_nodes', set())
                active_nodes_list = list(active_nodes_set)
                
                # 获取压缩配置（从训练参数中）
                enable_compression = data.get('enable_compression', False)
                compression_config = data.get('compression_config', {})
                
                result = run_federated_training(
                    config=config,
                    monitor_instance=monitor,
                    socketio_instance=socketio,
                    progress_callback=None,
                    active_nodes=active_nodes_list,
                    save_results=True,
                    results_dir='results',
                    enable_compression=enable_compression,
                    compression_config=compression_config
                )
                
                add_log('=' * 60, level='info')
                add_log('✅ 联邦训练完成！', level='info')
                add_log('=' * 60, level='info')
                
                # 训练完成
                monitor.end_experiment()
                training_state['experiment_running'] = False
                
                socketio.emit('update', {
                    'event_type': 'experiment_ended',
                    'data': {'timestamp': datetime.now().isoformat()},
                    'timestamp': datetime.now().isoformat()
                })
                socketio.emit('state_update', monitor.get_state())
                
            except Exception as e:
                add_log(f'训练过程出错: {str(e)}', level='error')
                monitor.end_experiment()
                training_state['experiment_running'] = False
                import traceback
                traceback.print_exc()
                socketio.emit('update', {
                    'event_type': 'experiment_error',
                    'data': {'error': str(e)},
                    'timestamp': datetime.now().isoformat()
                })
        
        # 在后台线程中启动训练
        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()
        
        add_log(f'训练已启动 (轮数: {rounds}, 批次大小: {batch_size}, 学习率: {learning_rate})')
        
        return jsonify({
            'success': True,
            'message': 'Training started successfully',
            'config': config
        })
    except Exception as e:
        add_log(f'启动训练失败: {str(e)}', level='error')
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/status', methods=['POST'])
def update_training_status():
    """接收训练状态更新（由训练脚本调用）"""
    data = request.get_json()
    action = data.get('action')
    
    try:
        if action == 'start':
            config = data.get('config', {})
            monitor.start_experiment(config)
            training_state['experiment_running'] = True
            socketio.emit('update', {
                'event_type': 'experiment_started',
                'data': {'config': config},
                'timestamp': datetime.now().isoformat()
            })
            
        elif action == 'round_started':
            round_num = data.get('round', 0)
            monitor.start_round(round_num)
            monitor.state['current_round'] = round_num
            socketio.emit('update', {
                'event_type': 'round_started',
                'data': {'round': round_num},
                'timestamp': datetime.now().isoformat()
            })
            
        elif action == 'round_metrics':
            round_num = data.get('round', 0)
            metrics = data.get('metrics', {})
            
            # 确保数组长度足够
            while len(monitor.state['global_metrics']['rounds']) <= round_num:
                monitor.state['global_metrics']['rounds'].append(len(monitor.state['global_metrics']['rounds']))
                monitor.state['global_metrics']['loss'].append(0.0)
                monitor.state['global_metrics']['f1'].append(0.0)
                monitor.state['global_metrics']['accuracy'].append(0.0)
            
            # 更新全局指标
            if 'loss' in metrics:
                monitor.state['global_metrics']['loss'][round_num] = metrics['loss']
            if 'f1' in metrics:
                monitor.state['global_metrics']['f1'][round_num] = metrics['f1']
            if 'accuracy' in metrics:
                monitor.state['global_metrics']['accuracy'][round_num] = metrics['accuracy']
            
            # 更新轮次指标
            monitor.update_round_metrics(round_num, {}, metrics)
            
            socketio.emit('update', {
                'event_type': 'round_metrics_updated',
                'data': {'round': round_num, 'global_metrics': metrics},
                'timestamp': datetime.now().isoformat()
            })
            
        elif action == 'node_status':
            node_id = data.get('node_id')
            status = data.get('status')
            node_metrics = data.get('metrics')
            monitor.update_node_status(node_id, status, metrics=node_metrics)
            
        elif action == 'end':
            monitor.end_experiment()
            training_state['experiment_running'] = False
            socketio.emit('update', {
                'event_type': 'experiment_ended',
                'data': {'timestamp': datetime.now().isoformat()},
                'timestamp': datetime.now().isoformat()
            })
        
        socketio.emit('state_update', monitor.get_state())
        return jsonify({'success': True})
        
    except Exception as e:
        add_log(f'更新训练状态失败: {str(e)}', level='error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """停止训练"""
    try:
        if training_state['experiment_running']:
            # 注意：由于训练在后台线程中运行，无法直接停止
            # 可以通过设置标志位来停止（需要在训练循环中检查）
            monitor.end_experiment()
            training_state['experiment_running'] = False
            add_log('训练已停止', level='info')
            
            socketio.emit('update', {
                'event_type': 'experiment_ended',
                'data': {'timestamp': datetime.now().isoformat()},
                'timestamp': datetime.now().isoformat()
            })
            socketio.emit('state_update', monitor.get_state())
            
            return jsonify({'success': True, 'message': 'Training stopped'})
        else:
            return jsonify({'error': 'No training running'}), 404
    except Exception as e:
        add_log(f'停止训练失败: {str(e)}', level='error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """获取日志"""
    limit = request.args.get('limit', 100, type=int)
    level = request.args.get('level', None)
    
    logs = training_state['logs']
    if level:
        logs = [log for log in logs if log.get('level') == level]
    
    return jsonify({'logs': logs[-limit:], 'total': len(logs)})


@app.route('/api/logs/clear', methods=['POST'])
def clear_logs():
    """清空日志"""
    training_state['logs'] = []
    return jsonify({'success': True, 'message': 'Logs cleared'})


@app.route('/api/analysis/data', methods=['GET'])
def get_analysis_data():
    """获取分析数据"""
    state = monitor.get_state()
    
    analysis = {
        'convergence': {
            'losses': state['global_metrics']['loss'],
            'f1_scores': state['global_metrics']['f1'],
            'accuracies': state['global_metrics']['accuracy'],
            'rounds': state['global_metrics']['rounds']
        },
        'node_performance': {},
        'training_time': {},
        'data_distribution': {}
    }
    
    # 节点性能分析
    for node_id, node_info in state['nodes'].items():
        metrics = node_info.get('metrics', {})
        analysis['node_performance'][node_id] = {
            'avg_loss': metrics.get('loss', 0),
            'avg_f1': metrics.get('f1', 0),
            'avg_accuracy': metrics.get('accuracy', 0),
            'data_size': node_info.get('data_size', 0)
        }
    
    # 训练时间分析
    for round_data in state['round_history']:
        round_num = round_data.get('round', 0)
        timestamp = round_data.get('timestamp', '')
        analysis['training_time'][round_num] = timestamp
    
    return jsonify(analysis)


@app.route('/api/analysis/convergence', methods=['GET'])
def get_convergence_analysis():
    """获取收敛分析"""
    state = monitor.get_state()
    losses = state['global_metrics']['loss']
    
    if len(losses) < 2:
        return jsonify({'error': 'Not enough data for convergence analysis'}), 400
    
    # 计算收敛指标
    recent_losses = losses[-10:] if len(losses) >= 10 else losses
    loss_change = abs(recent_losses[-1] - recent_losses[0])
    loss_std = sum((x - sum(recent_losses)/len(recent_losses))**2 for x in recent_losses) / len(recent_losses)
    loss_std = loss_std ** 0.5
    
    is_converged = loss_change < 0.001 and loss_std < 0.001
    
    return jsonify({
        'is_converged': is_converged,
        'loss_change': loss_change,
        'loss_std': loss_std,
        'convergence_rate': (losses[0] - losses[-1]) / len(losses) if len(losses) > 0 else 0,
        'total_rounds': len(losses)
    })


def add_log(message, level='info'):
    """添加日志"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'level': level,
        'message': message
    }
    training_state['logs'].append(log_entry)
    
    # 限制日志数量
    if len(training_state['logs']) > 1000:
        training_state['logs'] = training_state['logs'][-1000:]
    
    # 通过WebSocket广播日志
    socketio.emit('new_log', log_entry)


# 注意：MonitoredExperiment 类已不再使用（因为不再依赖 Fed-BioMed Experiment）
# 保留此类仅用于向后兼容，实际训练通过 federated_simulation_trainer.run_federated_training 进行


def run_visualization_server(host='0.0.0.0', port=5002, debug=False):
    """运行可视化服务器"""
    print("=" * 60)
    print(f"Starting Federated Learning Visualization Server")
    print(f"Template directory: {template_dir}")
    print(f"Static directory: {static_dir}")
    print(f"Server URL: http://{host}:{port}")
    print("=" * 60)
    print("Open your browser and navigate to the URL above to view the dashboard")
    print("")
    print("📋 使用说明:")
    print("   1. 在网页上激活节点（node_1, node_2, node_3）")
    print("   2. 节点激活后会准备好参与联邦训练")
    print("   3. 点击'开始训练'按钮启动联邦训练")
    print("   4. 训练将在单进程内模拟多客户端联邦学习")
    print("=" * 60)
    try:
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 运行服务器
    run_visualization_server(port=5002, debug=True)
