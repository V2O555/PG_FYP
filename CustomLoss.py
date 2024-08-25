import torch.nn.functional as F
from torch import nn


class CustomSparseCategoricalCrossentropy(nn.Module):
    def __init__(self, unlabeled=3):
        super(CustomSparseCategoricalCrossentropy, self).__init__()
        self.unlabeled = unlabeled

    def forward(self, y_pred, y_true):
        # 生成掩码，去掉未标记的部分
        haslabel = y_true != self.unlabeled  # [32, 1, 256, 256]

        # 去掉多余的维度，使掩码的形状与 y_true 一致
        haslabel = haslabel.squeeze(1)  # [32, 256, 256]

        # 展平掩码为一维
        haslabel_flat = haslabel.view(-1)  # [N * H * W]

        # 展平 y_true 和 y_pred
        y_true_flat = y_true.squeeze(1).view(-1)  # [N * H * W]
        y_pred_flat = y_pred.permute(0, 2, 3, 1).reshape(-1, y_pred.size(1))  # [N * H * W, C]

        # 使用展平掩码筛选 y_true 和 y_pred
        y_true_filtered = y_true_flat[haslabel_flat]  # [有效像素数]
        y_pred_filtered = y_pred_flat[haslabel_flat]  # [有效像素数, C]

        # 计算交叉熵损失
        return F.cross_entropy(y_pred_filtered, y_true_filtered.long(), reduction='mean')

