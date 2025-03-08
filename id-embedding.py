import torch
import torch.nn as nn
import torch.optim as optim

# 假设数据集
user_ids = torch.tensor([0, 1, 2, 3, 4])  # 用户 ID
item_ids = torch.tensor([0, 1, 2, 3, 4])  # 商品 ID
labels = torch.tensor([1, 0, 1, 0, 1])    # 标签（例如，点击与否）

# 定义模型
class IDEmbeddingModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(IDEmbeddingModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)  # 用户 ID 嵌入
        self.item_embedding = nn.Embedding(num_items, embedding_dim)  # 商品 ID 嵌入
        self.fc = nn.Linear(embedding_dim * 2, 1)  # 全连接层，输出预测结果
    
    def forward(self, user_ids, item_ids):
        # 获取用户和商品的嵌入向量
        user_emb = self.user_embedding(user_ids)  # Shape: (batch_size, embedding_dim)
        item_emb = self.item_embedding(item_ids)  # Shape: (batch_size, embedding_dim)
        
        # 拼接用户和商品的嵌入向量
        combined = torch.cat([user_emb, item_emb], dim=1)  # Shape: (batch_size, embedding_dim * 2)
        
        # 通过全连接层输出预测结果
        output = torch.sigmoid(self.fc(combined))  # Shape: (batch_size, 1)
        return output

# 超参数
num_users = 10  # 用户 ID 的数量
num_items = 10  # 商品 ID 的数量
embedding_dim = 8  # 嵌入维度
learning_rate = 0.01
num_epochs = 10

# 初始化模型、损失函数和优化器
model = IDEmbeddingModel(num_users, num_items, embedding_dim)
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(user_ids, item_ids).squeeze()  # 去掉多余的维度
    loss = criterion(outputs, labels.float())
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

with torch.no_grad():
    print(model(user_ids, item_ids))
    print(model.user_embedding(user_ids))
    print(model.item_embedding(item_ids))
