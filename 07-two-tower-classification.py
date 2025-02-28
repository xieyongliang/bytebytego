import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# 1. 定义双塔模型（Two-Tower Model）
# ---------------------------
class TwoTowerModel(nn.Module):
    def __init__(self, user_feature_dim, event_feature_dim, embedding_dim):
        super(TwoTowerModel, self).__init__()
        # 用户塔：将用户的原始特征映射到低维潜在空间
        self.user_encoder = nn.Sequential(
            nn.Linear(user_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        # 事件塔：将事件的内容特征（如元数据）映射到同一潜在空间
        self.event_encoder = nn.Sequential(
            nn.Linear(event_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
    
    def forward(self, user_features, event_features):
        # 得到用户和事件的嵌入
        user_embedding = self.user_encoder(user_features)    # (batch, embedding_dim)
        event_embedding = self.event_encoder(event_features)  # (batch, embedding_dim)
        # 点积作为评分
        scores = (user_embedding * event_embedding).sum(dim=1)
        return scores, user_embedding, event_embedding

# ---------------------------
# 2. 构造示例数据
# ---------------------------
num_users = 1000      # 用户数量
num_events = 500      # 事件数量
user_feature_dim = 10 # 用户特征维度（例如：年龄、性别、地区等）
event_feature_dim = 15# 事件特征维度（例如：类别、标签、地点、时间等）
embedding_dim = 32    # 潜在因子维度

# 随机生成用户和事件的特征（在实际场景中，这部分数据由内容特征提取获得）
np.random.seed(42)
user_features = np.random.rand(num_users, user_feature_dim).astype(np.float32)
event_features = np.random.rand(num_events, event_feature_dim).astype(np.float32)

# 构造用户-事件交互矩阵，假设1表示用户参与或点击过事件，0表示没有交互
# 对于冷启动用户或事件，交互记录可能为空，这时可利用基于内容的特征来初始化嵌入
interactions = np.random.randint(0, 2, size=(num_users, num_events)).astype(np.float32)

# ---------------------------
# 3. 构造PyTorch Dataset与DataLoader
# ---------------------------
class EventRecommendationDataset(Dataset):
    def __init__(self, user_features, event_features, interactions):
        self.user_features = torch.tensor(user_features)   # shape: (num_users, user_feature_dim)
        self.event_features = torch.tensor(event_features)   # shape: (num_events, event_feature_dim)
        self.interactions = torch.tensor(interactions)       # shape: (num_users, num_events)
        self.num_events = self.interactions.shape[1]
    
    def __len__(self):
        # 这里简单将所有用户与所有事件的组合作为训练样本
        return self.interactions.shape[0] * self.interactions.shape[1]
    
    def __getitem__(self, idx):
        # 将扁平索引转换成用户索引和事件索引
        user_idx = idx // self.num_events
        event_idx = idx % self.num_events
        user_feat = self.user_features[user_idx]
        event_feat = self.event_features[event_idx]
        label = self.interactions[user_idx, event_idx]
        return user_feat, event_feat, label

dataset = EventRecommendationDataset(user_features, event_features, interactions)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# ---------------------------
# 4. 训练模型
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoTowerModel(user_feature_dim, event_feature_dim, embedding_dim).to(device)
# 这里使用二分类交叉熵损失，因为标签为0/1（是否发生交互）
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    for user_batch, event_batch, labels in dataloader:
        user_batch = user_batch.to(device)
        event_batch = event_batch.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        scores, _, _ = model(user_batch, event_batch)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * user_batch.size(0)
    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# ---------------------------
# 5. 推理阶段：基于嵌入进行事件推荐
# ---------------------------
def recommend_events(user_idx, top_k=5):
    # 对于新用户或已有用户，根据用户的内容特征生成嵌入
    user_feat = torch.tensor(user_features[user_idx]).unsqueeze(0).to(device)
    with torch.no_grad():
        user_embedding = model.user_encoder(user_feat)  # shape: (1, embedding_dim)
    # 对于所有事件，根据其内容特征生成嵌入
    all_event_feats = torch.tensor(event_features).to(device)
    with torch.no_grad():
        event_embeddings = model.event_encoder(all_event_feats)  # shape: (num_events, embedding_dim)
    # 计算点积作为兴趣得分
    scores = (user_embedding * event_embeddings).sum(dim=1)
    top_scores, top_indices = torch.topk(scores, top_k)
    return top_indices.cpu().numpy(), top_scores.cpu().numpy()

# 示例：为用户0推荐5个事件
recommended_events, scores = recommend_events(0)
print("为用户 0 推荐的事件索引：", recommended_events)
print("对应的得分：", scores)
