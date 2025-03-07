import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ------------------------------
# 1. 定义房源嵌入网络
# ------------------------------
class ListingEmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        """
        input_dim: 输入特征维度
        embedding_dim: 输出嵌入维度
        """
        super(ListingEmbeddingNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
    
    def forward(self, x):
        x = self.model(x)
        # L2归一化，确保嵌入在单位球上，有助于距离计算
        x = F.normalize(x, p=2, dim=1)
        return x

# ------------------------------
# 2. 定义自定义损失函数
# ------------------------------
class CustomLTRLoss(nn.Module):
    def __init__(self, pos_margin=0.0, neg_margin=0.5,
                 weight_sliding_pos=1.0, weight_booking_pos=1.5, 
                 weight_random_neg=1.0, weight_hard_neg=1.5):
        """
        pos_margin: 正样本理想距离（通常为0）
        neg_margin: 负样本需要大于的最小距离
        各项权重：可根据任务调整
        """
        super(CustomLTRLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.w_sliding_pos = weight_sliding_pos
        self.w_booking_pos = weight_booking_pos
        self.w_random_neg = weight_random_neg
        self.w_hard_neg = weight_hard_neg

    def forward(self, center, sliding_pos, random_neg, booking_pos, hard_neg):
        """
        center: (B, d) 中心房源的嵌入
        sliding_pos: (B, N_pos, d) 滑动窗口正样本嵌入
        random_neg: (B, N_rand, d) 随机负样本嵌入（滑动窗口外）
        booking_pos: (B, N_booking, d) 最终预定正样本嵌入
        hard_neg: (B, N_hard, d) 同一区域但困难负样本嵌入
        
        对于正样本，目标是距离越小越好（理想为0），使用均方距离；对于负样本，使用 hinge 损失，
        使得距离大于 neg_margin，否则产生损失。
        """
        # 计算欧氏距离： (B, N, d) 与 (B, d) 的距离计算，需要扩展 center
        # 方法：中心向量扩展到 (B, N, d)，然后计算差的平方和再开方
        def calc_distance(center, samples):
            # center: (B, d) -> (B, 1, d)广播到 (B, N, d)
            diff = samples - center.unsqueeze(1)
            dist = torch.norm(diff, p=2, dim=2)  # (B, N)
            return dist
        
        # 正样本损失：滑动窗口正样本
        dist_sliding = calc_distance(center, sliding_pos)  # (B, N_pos)
        loss_sliding_pos = torch.mean(torch.pow(dist_sliding - self.pos_margin, 2))
        
        # 正样本损失：预定正样本
        dist_booking = calc_distance(center, booking_pos)  # (B, N_booking)
        loss_booking_pos = torch.mean(torch.pow(dist_booking - self.pos_margin, 2))
        
        # 负样本损失：随机负样本（滑动窗口外）
        dist_random = calc_distance(center, random_neg)  # (B, N_rand)
        loss_random_neg = torch.mean(torch.pow(torch.clamp(self.neg_margin - dist_random, min=0.0), 2))
        
        # 负样本损失：困难负样本
        dist_hard = calc_distance(center, hard_neg)  # (B, N_hard)
        loss_hard_neg = torch.mean(torch.pow(torch.clamp(self.neg_margin - dist_hard, min=0.0), 2))
        
        total_loss = (self.w_sliding_pos * loss_sliding_pos +
                      self.w_booking_pos * loss_booking_pos +
                      self.w_random_neg * loss_random_neg +
                      self.w_hard_neg * loss_hard_neg)
        return total_loss

# ------------------------------
# 3. 模拟数据与训练示例
# ------------------------------
if __name__ == "__main__":
    # 参数设置
    batch_size = 16
    input_dim = 50  # 房源的输入特征维度
    embedding_dim = 32

    # 构造房源嵌入网络
    embed_net = ListingEmbeddingNet(input_dim, embedding_dim)
    
    # 构造损失函数
    loss_fn = CustomLTRLoss(pos_margin=0.0, neg_margin=0.5,
                              weight_sliding_pos=1.0,
                              weight_booking_pos=1.5,
                              weight_random_neg=1.0,
                              weight_hard_neg=1.5)
    
    optimizer = optim.Adam(embed_net.parameters(), lr=0.001)
    
    # 模拟一批数据
    # center 房源： (B, input_dim)
    center_features = torch.randn(batch_size, input_dim)
    # 滑动窗口正样本：假设每个中心有 4 个正样本
    sliding_pos_features = torch.randn(batch_size, 4, input_dim)
    # 随机负样本：每个中心有 6 个随机负样本
    random_neg_features = torch.randn(batch_size, 6, input_dim)
    # 最终预定正样本：每个中心有 1 个预定正样本
    booking_pos_features = torch.randn(batch_size, 1, input_dim)
    # 困难负样本：每个中心有 3 个困难负样本
    hard_neg_features = torch.randn(batch_size, 3, input_dim)
    
    # 将原始特征通过嵌入网络获得嵌入
    center_emb = embed_net(center_features)  # (B, embedding_dim)
    # 对每个样本对批量reshape，将 (B, N, input_dim) 转换为 (B * N, input_dim)，再还原为 (B, N, embedding_dim)
    def embed_batch(x):
        B, N, D = x.shape
        x = x.view(B * N, D)
        emb = embed_net(x)
        emb = emb.view(B, N, -1)
        return emb
    
    sliding_pos_emb = embed_batch(sliding_pos_features)
    random_neg_emb = embed_batch(random_neg_features)
    booking_pos_emb = embed_batch(booking_pos_features)
    hard_neg_emb = embed_batch(hard_neg_features)
    
    # 前向传播并计算损失
    loss = loss_fn(center_emb, sliding_pos_emb, random_neg_emb, booking_pos_emb, hard_neg_emb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
