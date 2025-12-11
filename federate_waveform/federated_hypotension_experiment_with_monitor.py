# -*- coding: utf-8 -*-
"""
带监控的联邦学习实验脚本
集成可视化监控功能到联邦学习实验
"""

import os
import sys
import threading
import time
import socket as sock

# =============== 关键：在导入 fedbiomed 之前绑定 researcher 组件目录 ===============
# 让脚本和 CLI 初始化的 fbm-researcher 用同一套配置 / 证书
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESEARCHER_ROOT = os.path.join(BASE_DIR, "fbm-researcher")
os.environ["FBM_RESEARCHER_COMPONENT_ROOT"] = RESEARCHER_ROOT

# ================== Fed-BioMed 依赖检查 ==================
try:
    from fedbiomed.researcher.federated_workflows import Experiment
    from fedbiomed.researcher.aggregators.fedavg import FedAverage
    from fedbiomed.common.metrics import MetricTypes
except ImportError:
    print("=" * 60)
    print("错误: 未找到 fedbiomed 模块")
    print("=" * 60)
    print("请确保已激活正确的虚拟环境:")
    print("  source fb_env/bin/activate")
    print("")
    print("如果fedbiomed未安装，请运行:")
    print("  pip install fedbiomed[node, gui, researcher]")
    print("=" * 60)
    sys.exit(1)

# ================== 训练计划导入 ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from federated_hypotension_training_plan import HypotensionTrainingPlan

# ================== 监控相关导入 ==================
from federated_learning_visualization import (
    MonitoredExperiment,
    monitor,
    NODE_STATUS,
    run_visualization_server,  # 暂时未直接用，但保留
)


def is_port_available(port, host='0.0.0.0'):
    """检查端口是否可用"""
    try:
        test_sock = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
        test_sock.setsockopt(sock.SOL_SOCKET, sock.SO_REUSEADDR, 1)
        test_sock.bind((host, port))
        test_sock.close()
        return True
    except OSError:
        return False


def find_available_port(start_port=5000, max_attempts=10):
    """查找可用端口"""
    for i in range(max_attempts):
        port = start_port + i
        if is_port_available(port):
            return port
    return None


def run_experiment_with_monitoring(
    start_visualization=True,
    visualization_port=5000,
    visualization_host='0.0.0.0'
):
    """
    运行带监控的联邦学习实验
    """

    # ---------- 1. 可视化监控服务器 ----------
    if start_visualization:
        print("=" * 60)
        print("启动可视化监控服务器...")
        print("=" * 60)

        actual_port = visualization_port
        if not is_port_available(visualization_port, visualization_host):
            print(f"⚠️  端口 {visualization_port} 已被占用，尝试查找可用端口...")
            available_port = find_available_port(visualization_port)
            if available_port:
                actual_port = available_port
                print(f"✅ 找到可用端口: {actual_port}")
            else:
                print(f"❌ 无法找到可用端口，请手动指定其他端口")
                print(f"   使用 --port 参数，例如: --port 5001")
                return

        def start_server():
            try:
                from federated_learning_visualization import app, socketio
                socketio.run(
                    app,
                    host=visualization_host,
                    port=actual_port,
                    debug=False,
                    use_reloader=False
                )
            except OSError as e:
                if e.errno == 48:
                    print(f"\n⚠️  警告: 端口 {actual_port} 启动时被占用")
                else:
                    print(f"\n❌ 服务器启动错误: {e}")
            except Exception as e:
                print(f"\n❌ 服务器启动异常: {e}")

        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()

        time.sleep(2)
        try:
            test_sock = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
            test_host = 'localhost' if visualization_host == '0.0.0.0' else visualization_host
            result = test_sock.connect_ex((test_host, actual_port))
            test_sock.close()
            if result == 0:
                print(f"\n✅ 可视化服务器已启动!")
                print(f"📊 请在浏览器中打开: http://localhost:{actual_port}")
                if visualization_host != '0.0.0.0':
                    print(f"   或访问: http://{visualization_host}:{actual_port}")
            else:
                print(f"\n⚠️  警告: 无法确认服务器是否启动成功")
                print(f"   请检查端口 {actual_port} 是否可用")
        except Exception as e:
            print(f"\n⚠️  警告: 无法验证服务器状态: {e}")

        print("\n" + "=" * 60 + "\n")

    # ---------- 2. 模型 & 训练参数 ----------
    model_args = {
        'batch_size': 128,
        'data_path': 'federated_data/node_1/train.pth',  # 仅作占位，实际按 node/dataset 来
        'use_uci2': True,
        'uci2_base_dir': 'uci2_dataset'
    }

    training_args = {
        'loader_args': {
            'batch_size': 128,
        },
        'optimizer_args': {
            'lr': 4e-5
        },
        'epochs': 1,
        'dry_run': False,
        'batch_maxnum': 100,
        'test_ratio': 0.1,
        'test_metric': MetricTypes.F1_SCORE,
        'test_on_global_updates': True,
        'test_on_local_updates': True,
        'test_batch_size': 0,
        'shuffle_testing_dataset': False,
    }

    tags = ['#hypotension', '#waveform', '#ecg', '#uci2']
    rounds = 5

    experiment_config = {
        'round_limit': rounds,
        'tags': tags,
        'training_args': training_args,
        'model_args': model_args
    }

    monitor.start_experiment(experiment_config)

    print("=" * 60)
    print("创建 Fed-BioMed 实验")
    print("=" * 60)
    print(f"FBM_RESEARCHER_COMPONENT_ROOT = {os.environ.get('FBM_RESEARCHER_COMPONENT_ROOT')}")
    print(f"标签: {tags}")
    print(f"联邦轮数: {rounds}")
    print(f"每轮 epochs: {training_args['epochs']}")
    print(f"学习率: {training_args['optimizer_args']['lr']}")
    print("=" * 60)

    # ---------- 3. 创建 Experiment ----------
    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=HypotensionTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None
    )

    monitored_exp = MonitoredExperiment(exp, monitor)

    # ---------- 4. 运行实验 ----------
    try:
        print("\n开始联邦学习实验...")
        print("=" * 60)

        monitored_exp.run()

        print("\n保存训练好的模型...")
        model_save_path = './trained_hypotension_model'
        exp.training_plan().export_model(model_save_path)
        print(f"模型已保存到: {model_save_path}")

        print("\n" + "=" * 60)
        print("训练结果摘要")
        print("=" * 60)

        training_replies = exp.training_replies()
        print(f"\n完成的训练轮次: {list(training_replies.keys())}")

        for round_num in range(rounds):
            if round_num in training_replies:
                round_data = training_replies[round_num]
                print(f"\n第 {round_num + 1} 轮:")
                for node_id, reply in round_data.items():
                    print(f"  节点 {node_id}:")
                    print(f"    训练时间: {reply['timing']['rtime_training']:.2f}s")
                    print(f"    总时间: {reply['timing']['rtime_total']:.2f}s")
                    if 'test_results' in reply:
                        print(f"    测试结果: {reply['test_results']}")

        print("\n" + "=" * 60)
        print("实验成功完成!")
        print("=" * 60)
        print(f"\n📊 可视化监控页面仍在运行: http://localhost:{visualization_port}")
        print("   可以继续查看训练结果和指标")

    except Exception as e:
        print(f"\n❌ 实验出错: {e}")
        monitor.end_experiment()
        raise

    finally:
        monitor.end_experiment()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='运行带监控的联邦学习实验')
    parser.add_argument('--no-visualization', action='store_true',
                        help='不启动可视化服务器')
    parser.add_argument('--port', type=int, default=5000,
                        help='可视化服务器端口 (默认: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='可视化服务器主机 (默认: 0.0.0.0)')

    args = parser.parse_args()

    run_experiment_with_monitoring(
        start_visualization=not args.no_visualization,
        visualization_port=args.port,
        visualization_host=args.host
    )


if __name__ == '__main__':
    main()
