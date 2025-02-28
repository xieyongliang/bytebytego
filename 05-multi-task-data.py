import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MultiTaskDataset(Dataset):
    def __init__(self, input_data, task1_labels, task2_labels):
        """
        dataset: 原始数据集（如 ImageFolder 或其他自定义数据集）
        transform: 数据增强方法（如 transform_train）
        """
        self.input_data = input_data
        self.task1_labels = task1_labels.squeeze()
        self.task2_labels = task2_labels.squeeze()

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        data = self.input_data[index]
        label1 = self.task1_labels[index]
        label2 = self.task2_labels[index]
        return data, label1, label2

# 定义Multi-Task Neural Network
class MultiTaskNN(nn.Module):
    def __init__(self, input_size, shared_size, task1_size, task2_size):
        super(MultiTaskNN, self).__init__()

        # 共享层
        self.shared_layer = nn.Sequential(
            nn.Linear(input_size, shared_size),
            nn.ReLU(),
            nn.Linear(shared_size, shared_size),
            nn.ReLU()
        )
        
        # 任务1的特定头部
        self.task1_head = nn.Sequential(
            nn.Linear(shared_size, task1_size),
            nn.ReLU(),
            nn.Linear(task1_size, 1)  # 二分类任务输出1个神经元
        )
        
        # 任务2的特定头部
        self.task2_head = nn.Sequential(
            nn.Linear(shared_size, task2_size),
            nn.ReLU(),
            nn.Linear(task2_size, 1)  # 二分类任务输出1个神经元
        )

    def forward(self, x):
        # 共享特征
        shared_features = self.shared_layer(x)
        
        # 任务1的输出
        task1_output = torch.sigmoid(self.task1_head(shared_features))
        
        # 任务2的输出
        task2_output = torch.sigmoid(self.task2_head(shared_features))
        
        return task1_output, task2_output

if __name__ == '__main__':
    # 模拟输入数据
    input_size = 100   # 输入特征的大小
    shared_size = 64   # 共享层特征数
    task1_size = 32    # 任务1头部神经元数
    task2_size = 32    # 任务2头部神经元数
    num_epochs = 10
    device = 'cpu'

    model = MultiTaskNN(input_size, shared_size, task1_size, task2_size)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_data = torch.randn(32, input_size)  # 32个样本，每个样本100个特征
    task1_labels = torch.randint(0, 2, (32, 1)).float()  # 任务1标签
    task2_labels = torch.randint(0, 2, (32, 1)).float()  # 任务2标签

    multitask_dataset = MultiTaskDataset(input_data, task1_labels, task2_labels)
    train_loader = DataLoader(multitask_dataset, batch_size=32, shuffle=True, num_workers=4)

    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()

        for data, label1, label2 in train_loader:
            data, label1, label2 = data.to(device), label1.to(device), label2.to(device)
            task1_output, task2_output = model(input_data)

            # 计算任务1和任务2的损失
            task1_loss = criterion(task1_output, task1_labels)
            task2_loss = criterion(task2_output, task2_labels)

            # 计算总损失（可以给不同任务的损失赋予不同权重）
            loss = task1_loss + task2_loss

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Task 1 Loss: {task1_loss.item()}, Task 2 Loss: {task2_loss.item()}, Loss: {loss.item()}")

