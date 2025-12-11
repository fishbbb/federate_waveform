#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动可视化服务器的便捷脚本
"""

import os
import sys

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入可视化模块
try:
    from federated_learning_visualization import run_visualization_server
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的目录下运行此脚本")
    sys.exit(1)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='启动联邦学习可视化服务器')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=5002, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("联邦学习可视化服务器")
    print("=" * 60)
    
    try:
        run_visualization_server(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"\n服务器启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
