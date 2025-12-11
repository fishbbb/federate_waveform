# -*- coding: utf-8 -*-
"""
Fed-BioMed Experiment Script for Hypotension Prediction
This script demonstrates how to run federated learning for hypotension prediction
"""

import os
import sys
import json
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.common.metrics import MetricTypes

# Import the training plan
# 确保从当前目录导入
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from federated_hypotension_training_plan import HypotensionTrainingPlan

# 尝试导入状态报告器
try:
    from training_status_reporter import get_reporter
    reporter = get_reporter()
    USE_REPORTER = True
except ImportError:
    USE_REPORTER = False
    reporter = None

def main():
    """
    Main function to run federated learning experiment
    """
    
    # 检查是否有配置从环境变量传入
    config_json = os.environ.get('FL_TRAINING_CONFIG')
    if config_json and os.path.exists(config_json):
        with open(config_json, 'r') as f:
            external_config = json.load(f)
        
        # 使用外部配置
        model_args = external_config.get('model_args', {
            'batch_size': 128,
            'data_path': 'federated_data/node_1/train.pth',
            'use_uci2': True,
            'uci2_base_dir': 'uci2_dataset'
        })
        training_args = external_config.get('training_args', {
            'loader_args': {'batch_size': 128},
            'optimizer_args': {'lr': 4e-5},
            'epochs': 1
        })
        rounds = external_config.get('round_limit', 5)
        tags = external_config.get('tags', ['#hypotension', '#waveform', '#ecg', '#uci2'])
        
        # 报告实验开始
        if USE_REPORTER:
            reporter.report_experiment_started(external_config)
    else:
        # 使用默认配置
        model_args = {
            'batch_size': 128,
            'data_path': 'federated_data/node_1/train.pth',
            'use_uci2': True,
            'uci2_base_dir': 'uci2_dataset'
        }
        training_args = {
            'loader_args': {'batch_size': 128},
            'optimizer_args': {'lr': 4e-5},
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
    
    # Create experiment
    print("=" * 60)
    print("Creating Fed-BioMed Experiment for Hypotension Prediction")
    print("=" * 60)
    print(f"Tags: {tags}")
    print(f"Rounds: {rounds}")
    print(f"Epochs per round: {training_args['epochs']}")
    print(f"Learning rate: {training_args['optimizer_args']['lr']}")
    print("=" * 60)
    
    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=HypotensionTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None  # Use all available nodes
    )
    
    # Run the experiment with monitoring
    print("\nStarting federated learning experiment...")
    
    # 创建自定义的run方法来监控训练过程
    class MonitoredExperimentWrapper:
        def __init__(self, exp):
            self.exp = exp
            self.original_run = exp.run
        
        def run(self):
            """运行实验并报告状态"""
            # 运行原始实验
            self.exp.run()
            
            # 处理训练结果并报告
            training_replies = self.exp.training_replies()
            
            for round_num in range(rounds):
                if round_num in training_replies:
                    # 报告轮次开始
                    if USE_REPORTER:
                        reporter.report_round_started(round_num)
                    
                    round_data = training_replies[round_num]
                    
                    # 处理每个节点的结果
                    node_metrics = {}
                    global_loss = 0
                    global_f1 = 0
                    global_acc = 0
                    node_count = 0
                    
                    for node_id, reply in round_data.items():
                        # 报告节点状态
                        if USE_REPORTER:
                            reporter.report_node_status(
                                node_id,
                                'completed',
                                metrics={
                                    'loss': reply.get('loss', 0),
                                    'f1': reply.get('test_results', {}).get('f1_score', 0),
                                    'accuracy': reply.get('test_results', {}).get('accuracy', 0),
                                    'training_time': reply.get('timing', {}).get('rtime_training', 0)
                                }
                            )
                        
                        node_metrics[node_id] = {
                            'loss': reply.get('loss', 0),
                            'f1': reply.get('test_results', {}).get('f1_score', 0),
                            'accuracy': reply.get('test_results', {}).get('accuracy', 0)
                        }
                        
                        global_loss += reply.get('loss', 0)
                        global_f1 += reply.get('test_results', {}).get('f1_score', 0)
                        global_acc += reply.get('test_results', {}).get('accuracy', 0)
                        node_count += 1
                    
                    # 计算全局指标
                    if node_count > 0:
                        global_metrics = {
                            'loss': global_loss / node_count,
                            'f1': global_f1 / node_count,
                            'accuracy': global_acc / node_count
                        }
                        
                        # 报告轮次指标
                        if USE_REPORTER:
                            reporter.report_round_metrics(round_num, global_metrics)
                        
                        print(f"\nRound {round_num + 1} - Global Metrics:")
                        print(f"  Loss: {global_metrics['loss']:.4f}")
                        print(f"  F1: {global_metrics['f1']:.4f}")
                        print(f"  Accuracy: {global_metrics['accuracy']:.4f}")
    
    # 包装实验对象
    monitored_exp = MonitoredExperimentWrapper(exp)
    monitored_exp.run()
    
    # 报告实验结束
    if USE_REPORTER:
        reporter.report_experiment_ended()
    
    # Save the trained model
    print("\nSaving trained model...")
    model_save_path = './trained_hypotension_model'
    exp.training_plan().export_model(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Display training results
    print("\n" + "=" * 60)
    print("Training Results Summary")
    print("=" * 60)
    
    print(f"\nTraining rounds completed: {list(exp.training_replies().keys())}")
    
    # Display results for each round
    for round_num in range(rounds):
        if round_num in exp.training_replies():
            round_data = exp.training_replies()[round_num]
            print(f"\nRound {round_num + 1}:")
            for node_id, reply in round_data.items():
                print(f"  Node {node_id}:")
                print(f"    Training time: {reply['timing']['rtime_training']:.2f}s")
                print(f"    Total time: {reply['timing']['rtime_total']:.2f}s")
                if 'test_results' in reply:
                    print(f"    Test results: {reply['test_results']}")
    
    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

