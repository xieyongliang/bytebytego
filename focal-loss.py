import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        :param alpha: 加权因子，用于调整不同类别的损失权重
        :param gamma: 调节因子，控制难易样本的关注程度
        :param reduction: 损失的聚合方式，可以是 'none'、'mean' 或 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: 模型的预测输出，概率值，形状为 (batch_size, num_classes)
        :param targets: 标签，形状为 (batch_size) 或 (batch_size, num_classes)
        """
        # 使用交叉熵损失
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        log_probs = torch.log_softmax(inputs, dim=1)
        loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        print(BCE_loss, log_probs, loss)
        
        # 获取预测的概率值
        pt = torch.exp(-BCE_loss)
        
        # 计算 Focal Loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# 示例用法
if __name__ == "__main__":
    # 假设有 3 类的分类任务
    inputs = torch.tensor([[0.5, 0.2, 0.3], [0.1, 0.6, 0.3]], dtype=torch.float32)  # 模型预测
    targets = torch.tensor([0, 1], dtype=torch.long)  # 真实标签

    focal_loss = FocalLoss(alpha=1, gamma=2)
    loss = focal_loss(inputs, targets)
    print("Focal Loss:", loss.item())
