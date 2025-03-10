import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# 1. 构造示例数据
# ------------------------------
# 假设有1000个房源，每个房源有20个特征（例如面积、价格、地理位置编码等）
num_listings = 1000
feature_dim = 20
np.random.seed(42)
listings_features = np.random.rand(num_listings, feature_dim).astype(np.float32)

# 构造训练数据：构造房源对及其相似度标签
# 这里简单地假设两个房源如果欧氏距离较小，则它们是相似的
def compute_similarity(feat1, feat2, threshold=0.5):
    # 计算欧氏距离，若小于阈值则认为相似（标签1），否则不相似（标签0）
    dist = np.linalg.norm(feat1 - feat2)
    return 1.0 if dist < threshold else 0.0

num_pairs = 5000
pairs = []
labels = []
for _ in range(num_pairs):
    i = np.random.randint(0, num_listings)
    j = np.random.randint(0, num_listings)
    label = compute_similarity(listings_features[i], listings_features[j], threshold=0.5)
    pairs.append((i, j))
    labels.append(label)
pairs = np.array(pairs)
labels = np.array(labels).astype(np.float32)

# 定义Dataset，返回两个房源的特征和标签
class ListingPairDataset(Dataset):
    def __init__(self, pairs, labels, listings_features):
        self.pairs = pairs
        self.labels = labels
        self.listings_features = listings_features
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        feat1 = self.listings_features[i]
        feat2 = self.listings_features[j]
        label = self.labels[idx]
        return feat1, feat2, label

dataset = ListingPairDataset(pairs, labels, listings_features)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ------------------------------
# 2. 定义浅层神经网络模型
# ------------------------------
class SimilarityModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32):
        """
        输入为两个房源特征向量的拼接，输出一个标量（0~1），表示相似度概率
        """
        super(SimilarityModel, self).__init__()
        self.fc1 = nn.Linear(2 * feature_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, feat1, feat2):
        # 拼接两个房源的特征向量
        x = torch.cat([feat1, feat2], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

model = SimilarityModel(feature_dim=feature_dim, hidden_dim=32)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------
# 3. 训练模型
# ------------------------------
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for feat1, feat2, label in dataloader:
        label = label.unsqueeze(1)  # shape (batch, 1)
        optimizer.zero_grad()
        output = model(feat1, feat2)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * feat1.size(0)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataset):.4f}")

# ------------------------------
# 4. 推荐系统：给定一个房源，推荐相似的房源
# ------------------------------
def recommend_similar(query_idx, listings_features, model, top_k=5):
    """
    对于给定房源 query_idx，计算其与所有其他房源的相似度得分，
    并返回得分最高的 top_k 个房源索引和得分。
    """
    model.eval()
    query_feat = torch.tensor(listings_features[query_idx]).unsqueeze(0)  # (1, feature_dim)
    scores = []
    with torch.no_grad():
        for idx in range(len(listings_features)):
            if idx == query_idx:
                continue
            candidate_feat = torch.tensor(listings_features[idx]).unsqueeze(0)
            score = model(query_feat, candidate_feat).item()
            scores.append((idx, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# 示例：为房源 0 推荐相似的房源
recommendations = recommend_similar(query_idx=0, listings_features=listings_features, model=model, top_k=5)
print("Recommendations for listing 0:")
for idx, score in recommendations:
    print(f"Listing {idx} with similarity score: {score:.4f}")
