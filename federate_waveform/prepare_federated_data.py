# -*- coding: utf-8 -*-
"""
Data Preparation Script for Federated Learning with uci2_dataset
This script splits the uci2_dataset into multiple parts for different nodes
"""

import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path


def create_hypotension_label(sp, dp, threshold_sp=90, threshold_dp=60):
    """
    根据SP和DP创建低血压标签
    
    Args:
        sp: 收缩压值
        dp: 舒张压值
        threshold_sp: 收缩压低血压阈值（默认90 mmHg）
        threshold_dp: 舒张压低血压阈值（默认60 mmHg）
    
    Returns:
        低血压标签 (1=低血压, 0=正常)
    """
    return ((sp < threshold_sp) | (dp < threshold_dp)).astype(int)


def load_uci2_data(base_dir='uci2_dataset', folds=[0, 1, 2], use_csv=True):
    """
    加载uci2_dataset数据
    
    Args:
        base_dir: uci2_dataset文件夹路径
        folds: 要加载的fold列表
        use_csv: 如果True使用CSV文件，如果False使用MAT文件
    
    Returns:
        DataFrame: 包含所有fold数据的DataFrame
    """
    all_dataframes = []
    
    for fold in folds:
        if use_csv:
            csv_path = os.path.join(base_dir, f'feat_fold_{fold}.csv')
            if not os.path.exists(csv_path):
                print(f"Warning: CSV file not found: {csv_path}, skipping fold {fold}")
                continue
            
            print(f"Loading CSV file: {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            # 如果使用MAT文件，需要从prepare_uci2_data.py导入相关逻辑
            from scipy.io import loadmat
            mat_path = os.path.join(base_dir, f'signal_fold_{fold}.mat')
            if not os.path.exists(mat_path):
                print(f"Warning: MAT file not found: {mat_path}, skipping fold {fold}")
                continue
            
            print(f"Loading MAT file: {mat_path}")
            mat_data = loadmat(mat_path)
            
            # Extract data
            patient = mat_data['patient'].flatten()
            trial = mat_data['trial'].flatten()
            sp = mat_data['SP'].flatten()
            dp = mat_data['DP'].flatten()
            
            # Create DataFrame
            df = pd.DataFrame({
                'patient': [str(p) for p in patient],
                'trial': [str(t) for t in trial],
                'SP': sp,
                'DP': dp
            })
        
        # 创建ID
        df['id'] = df['patient'].astype(str) + '_' + df['trial'].astype(str)
        
        # 创建低血压标签
        df['label'] = create_hypotension_label(
            df['SP'].values,
            df['DP'].values,
            threshold_sp=90,
            threshold_dp=60
        )
        
        print(f"  Fold {fold}: {len(df)} samples")
        print(f"    Hypotension: {df['label'].sum()} ({df['label'].sum()/len(df)*100:.2f}%)")
        print(f"    Normal: {(df['label']==0).sum()} ({(df['label']==0).sum()/len(df)*100:.2f}%)")
        
        all_dataframes.append(df)
    
    # 合并所有fold的数据
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"\nTotal combined samples: {len(combined_df)}")
        print(f"  Hypotension: {combined_df['label'].sum()} ({combined_df['label'].sum()/len(combined_df)*100:.2f}%)")
        print(f"  Normal: {(combined_df['label']==0).sum()} ({(combined_df['label']==0).sum()/len(combined_df)*100:.2f}%)")
        return combined_df
    else:
        raise ValueError("No data loaded from any fold")


def split_data_for_federated_learning(
    base_dir='uci2_dataset',
    output_dir='federated_data',
    num_nodes=3,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42,
    folds=[0, 1, 2],
    use_csv=True,
    merge_folds=True
):
    """
    Split uci2_dataset into multiple parts for federated learning nodes
    
    Args:
        base_dir: uci2_dataset文件夹路径
        output_dir: Directory to save split data
        num_nodes: Number of federated learning nodes
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_seed: Random seed for reproducibility
        folds: List of folds to use (0, 1, 2)
        use_csv: If True use CSV files, if False use MAT files
        merge_folds: If True merge all folds, if False process each fold separately
    """
    
    np.random.seed(random_seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if merge_folds:
        # 合并所有fold的数据
        print("=" * 80)
        print("合并所有fold的数据用于联邦学习")
        print("=" * 80)
        
        source_data = load_uci2_data(base_dir, folds, use_csv)
        
        # Get all IDs
        id_list = source_data['id'].tolist()
        print(f"\nTotal samples: {len(id_list)}")
        
        # Shuffle IDs
        np.random.shuffle(id_list)
        
        # Split into train/val/test
        n_total = len(id_list)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_ids = id_list[:n_train]
        val_ids = id_list[n_train:n_train + n_val]
        test_ids = id_list[n_train + n_val:]
        
        print(f"\nData split:")
        print(f"  Train: {len(train_ids)} ({len(train_ids)/n_total*100:.2f}%)")
        print(f"  Val: {len(val_ids)} ({len(val_ids)/n_total*100:.2f}%)")
        print(f"  Test: {len(test_ids)} ({len(test_ids)/n_total*100:.2f}%)")
        
        # Split training data for each node
        train_ids_per_node = np.array_split(train_ids, num_nodes)
        
        # Save data for each node
        for node_idx in range(num_nodes):
            node_dir = os.path.join(output_dir, f'node_{node_idx + 1}')
            os.makedirs(node_dir, exist_ok=True)
            
            # Get training IDs for this node
            node_train_ids = train_ids_per_node[node_idx].tolist()
            
            # Create split dictionary
            split_dict = {
                'train': node_train_ids,
                'val': val_ids,  # All nodes use same validation set
            }
            
            # Save split file
            split_path = os.path.join(node_dir, 'train.pth')
            torch.save(split_dict, split_path)
            print(f"\nNode {node_idx + 1}: Saved split to {split_path}")
            print(f"  Train samples: {len(node_train_ids)}")
            print(f"  Val samples: {len(val_ids)}")
            
            # Save CSV file reference (保存数据路径信息)
            node_info_path = os.path.join(node_dir, 'data_info.txt')
            with open(node_info_path, 'w') as f:
                f.write(f"base_dir={base_dir}\n")
                f.write(f"use_csv={use_csv}\n")
                f.write(f"folds={folds}\n")
            print(f"  Saved data info to: {node_info_path}")
        
        # Save test split separately
        test_split = {'test': test_ids}
        test_path = os.path.join(output_dir, 'test.pth')
        torch.save(test_split, test_path)
        print(f"\nTest split saved to: {test_path}")
        print(f"Test samples: {len(test_ids)}")
        
        # Save summary
        summary = {
            'total_samples': n_total,
            'num_nodes': num_nodes,
            'train_per_node': [len(ids) for ids in train_ids_per_node],
            'val_samples': len(val_ids),
            'test_samples': len(test_ids),
            'folds_used': folds,
            'use_csv': use_csv
        }
        
        summary_path = os.path.join(output_dir, 'split_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Federated Learning Data Split Summary (uci2_dataset)\n")
            f.write("=" * 50 + "\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\nSummary saved to: {summary_path}")
        print("\nData preparation completed!")
        
    else:
        # 为每个fold分别创建分割（不常用）
        print("=" * 80)
        print("为每个fold分别创建分割")
        print("=" * 80)
        # 这里可以实现分别处理的逻辑，但通常合并所有fold更常用
        raise NotImplementedError("分别处理每个fold的功能暂未实现，请使用merge_folds=True")


def main():
    """
    Main function to prepare data for federated learning
    """
    # Paths - adjust these according to your setup
    # 从federate_waveform文件夹向上到根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    uci2_dataset_dir = os.path.join(base_dir, 'uci2_dataset')
    output_dir = os.path.join(base_dir, 'federated_data')
    
    # Check if uci2_dataset directory exists
    if not os.path.exists(uci2_dataset_dir):
        print(f"Error: uci2_dataset directory not found at: {uci2_dataset_dir}")
        print("Please check the path and try again.")
        return
    
    # Check if CSV files exist
    csv_files_exist = all(
        os.path.exists(os.path.join(uci2_dataset_dir, f'feat_fold_{fold}.csv'))
        for fold in [0, 1, 2]
    )
    
    if not csv_files_exist:
        print(f"Warning: Some CSV files are missing in {uci2_dataset_dir}")
        print("Please ensure feat_fold_0.csv, feat_fold_1.csv, and feat_fold_2.csv exist")
    
    # Split data for 3 nodes (you can change this)
    split_data_for_federated_learning(
        base_dir=uci2_dataset_dir,
        output_dir=output_dir,
        num_nodes=3,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
        folds=[0, 1, 2],  # 使用所有fold
        use_csv=True,  # 使用CSV特征文件
        merge_folds=True  # 合并所有fold
    )


if __name__ == '__main__':
    main()
