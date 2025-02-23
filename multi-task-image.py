import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultiTaskNetwork(nn.Module):
    def __init__(self, num_classes_task1=10, num_classes_task2=5):
        super(MultiTaskNetwork, self).__init__()
        
        # 共享的特征提取层（CNN）
        self.shared_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 输入通道 3，输出通道 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 输入通道 32，输出通道 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 输入通道 64，输出通道 128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 任务 1 的分支（物体类别分类）
        self.task1_fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),  # 假设输入图像大小为 32x32，经过 3 次池化后为 4x4
            nn.ReLU(),
            nn.Linear(256, num_classes_task1)
        )
        
        # 任务 2 的分支（颜色类别分类）
        self.task2_fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),  # 同上
            nn.ReLU(),
            nn.Linear(256, num_classes_task2)
        )
    
    def forward(self, x):
        # 共享特征提取
        x = self.shared_layers(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        
        # 任务 1 的输出
        output_task1 = self.task1_fc(x)
        
        # 任务 2 的输出
        output_task2 = self.task2_fc(x)
        
        return output_task1, output_task2

'''
# 示例用法
if __name__ == "__main__":
    # 创建模型
    model = MultiTaskNetwork(num_classes_task1=10, num_classes_task2=5)
    
    # 示例输入（batch_size=8, 3 通道, 32x32 图像）
    input_tensor = torch.randn(8, 3, 32, 32)
    
    # 前向传播
    output_task1, output_task2 = model(input_tensor)
    
    print("任务 1 的输出形状：", output_task1.shape)  # 应为 (8, 10)
    print("任务 2 的输出形状：", output_task2.shape)  # 应为 (8, 5)
'''

# 创建模型、优化器和损失函数
model = MultiTaskNetwork(num_classes_task1=10, num_classes_task2=5)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion_task1 = nn.CrossEntropyLoss()  # 任务 1 的损失函数
criterion_task2 = nn.CrossEntropyLoss()  # 任务 2 的损失函数

# 示例训练循环
for epoch in range(10):  # 训练 10 个 epoch
    # 示例输入和标签
    inputs = torch.randn(8, 3, 32, 32)  # 输入图像
    labels_task1 = torch.randint(0, 10, (8,))  # 任务 1 的标签
    labels_task2 = torch.randint(0, 5, (8,))   # 任务 2 的标签
    
    # 前向传播
    outputs_task1, outputs_task2 = model(inputs)
    
    # 计算损失
    loss_task1 = criterion_task1(outputs_task1, labels_task1)
    loss_task2 = criterion_task2(outputs_task2, labels_task2)
    total_loss = loss_task1 + loss_task2  # 总损失
    
    # 反向传播和优化
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/10], Loss: {total_loss.item()}")
