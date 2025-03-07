import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# ------------------------------
# 1. 定义GCN模型，用于学习节点嵌入
# ------------------------------
class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels, embedding_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_dim)
    
    def forward(self, x, edge_index):
        # 第一层卷积 + ReLU 激活
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二层卷积得到最终节点嵌入
        x = self.conv2(x, edge_index)
        return x

# ------------------------------
# 2. 定义链路预测模型
# ------------------------------
class LinkPredictionModel(nn.Module):
    def __init__(self, gcn):
        """
        :param gcn: 用于生成节点嵌入的GCN模型
        """
        super(LinkPredictionModel, self).__init__()
        self.gcn = gcn
    
    def forward(self, x, edge_index, edge_pairs):
        """
        :param x: 时刻 t 的节点特征, shape: (num_nodes, num_features)
        :param edge_index: 时刻 t 的图连接，shape: (2, num_edges)
        :param edge_pairs: 待预测边（节点对），shape: (batch_size, 2)
        :return: 对每个边预测一个概率，表示时刻 t+1 边是否存在
        """
        # 计算节点嵌入
        embeddings = self.gcn(x, edge_index)  # (num_nodes, embedding_dim)
        
        # 对每个候选边 (i, j) 计算点积作为打分
        i_nodes = edge_pairs[:, 0]
        j_nodes = edge_pairs[:, 1]
        scores = (embeddings[i_nodes] * embeddings[j_nodes]).sum(dim=1)
        
        # 使用 sigmoid 将分数映射到概率 (0,1)
        probs = torch.sigmoid(scores)
        return probs

# ------------------------------
# 3. 构造示例数据
# ------------------------------
# 假设我们有一个包含100个节点的图，时刻 t 的节点特征为随机数，特征维度为16
num_nodes = 100
num_features = 16
x = torch.randn((num_nodes, num_features))

# 构造时刻 t 的边（例如随机生成一些边）
# edge_index 的形状为 (2, num_edges)
num_edges = 500
edge_index = torch.randint(0, num_nodes, (2, num_edges))

# 构造训练样本：这些是时刻 t+1 的候选边及标签（1表示边存在，0表示不存在）
# 例如，随机选取200个边候选，并随机生成标签（在实际中应该来自真实数据）
num_train = 200
edge_pairs = torch.randint(0, num_nodes, (num_train, 2))  # 每行是 (i, j)
labels = torch.randint(0, 2, (num_train,)).float()         # 标签：0或1

# ------------------------------
# 4. 训练模型
# ------------------------------
# 参数设置：GCN输入特征维度=num_features, 隐藏层32维, 嵌入输出16维
embedding_dim = 16
gcn = GCN(num_features=num_features, hidden_channels=32, embedding_dim=embedding_dim)
model = LinkPredictionModel(gcn)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

model.train()
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    preds = model(x, edge_index, edge_pairs)  # 预测时刻 t+1 边的存在概率
    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# ------------------------------
# 5. 推理：为给定节点推荐可能的新连接 (PYMK)
# ------------------------------
def recommend_new_edges(target_node, x, edge_index, model, existing_edges, top_k=5):
    """
    对于给定节点 target_node，计算其与所有其他节点的预测链接概率，
    并排除已有连接，推荐 top_k 个最可能的新连接。
    
    :param target_node: int，待推荐的目标节点
    :param x: 时刻 t 的节点特征
    :param edge_index: 时刻 t 的边连接
    :param model: 链路预测模型
    :param existing_edges: set，包含已有连接 (i, j) 的集合（无向）
    :param top_k: 推荐数目
    :return: 推荐的节点列表及对应概率
    """
    model.eval()
    with torch.no_grad():
        # 生成目标节点与所有其他节点的候选边 (target_node, i)
        candidate_nodes = torch.arange(0, num_nodes)
        # 排除自身
        mask = candidate_nodes != target_node
        candidate_nodes = candidate_nodes[mask]
        # 构造候选边对
        candidate_pairs = torch.stack([torch.full_like(candidate_nodes, target_node), candidate_nodes], dim=1)
        # 预测连接概率
        probs = model(x, edge_index, candidate_pairs)
        
        # 排除已经存在的连接
        filtered = []
        for i, prob in zip(candidate_nodes.tolist(), probs.tolist()):
            if (target_node, i) not in existing_edges and (i, target_node) not in existing_edges:
                filtered.append((i, prob))
        
        # 根据概率排序，选取 top_k
        filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]
        return filtered

# 假设已有的边以元组形式存储在集合 existing_edges 中
existing_edges = set()
src, dst = edge_index.tolist()
for s, d in zip(src, dst):
    existing_edges.add((s, d))
    existing_edges.add((d, s))  # 假设无向

# 为节点 0 推荐新连接
recommendations = recommend_new_edges(target_node=0, x=x, edge_index=edge_index, model=model, existing_edges=existing_edges, top_k=5)
print("为节点0推荐的新连接（节点，预测概率）：")
print(recommendations)
