import torch
import numpy as np
import time, os
from sklearn.metrics import f1_score, accuracy_score
from torch import nn


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calc_f1(y_true, y_pre, threshold=0.5):
    """
    y_true: torch.Tensor, 0/1
    y_pre : torch.Tensor, 概率或logits经过sigmoid后的概率
    """
    y_true = y_true.view(-1).detach().cpu().numpy().astype(int)         # np.int -> int
    y_pred = (y_pre.view(-1).detach().cpu().numpy() >= threshold).astype(int)
    return f1_score(y_true, y_pred), accuracy_score(y_true, y_pred)


def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class Loss_cal(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(Loss_cal, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        # 保证 targets 与 outputs 的 dtype/shape 一致（BCEWithLogitsLoss 需要 float）
        targets = targets.to(dtype=outputs.dtype).view_as(outputs)
        loss = self.criterion(outputs, targets)
        pt = torch.exp(-loss)  # = sigmoid logits 后的概率对应的损失转化
        focal = self.alpha * (1 - pt) ** self.gamma * loss
        return focal.mean()
