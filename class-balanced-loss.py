import torch
import torch.nn as nn
import numpy as np

class ClassBalancedLoss(nn.Module):
    def __init__(self, beta, num_classes):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, labels, samples_per_class):
        # 计算每个类别的权重
        effective_num = 1.0 - np.power(self.beta, samples_per_class)
        class_weights = (1.0 - self.beta) / np.array(effective_num)
        class_weights = class_weights / np.sum(class_weights) * self.num_classes
        
        # 转换为 tensor
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(logits.device)
        
        # 计算交叉熵损失
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        loss = ce_loss(logits, labels)
        
        return loss

# 示例使用
beta = 0.999
num_classes = 10
samples_per_class = [100, 500, 1000, 50, 200, 300, 10, 20, 80, 60]  # 每个类别的样本数

loss_fn = ClassBalancedLoss(beta, num_classes)

logits = torch.randn(32, num_classes)  # 假设32个样本，logits的输出
labels = torch.randint(0, num_classes, (32,))  # 32个样本的标签

loss = loss_fn(logits, labels, samples_per_class)
print("Loss:", loss.item())
