import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ------------------------------
# 定义多任务网络
# ------------------------------
class MultiTaskNewsFeedModel(nn.Module):
    def __init__(self, input_dim, shared_hidden_dim, cls_hidden_dim, reg_hidden_dim):
        """
        :param input_dim: 输入特征维度
        :param shared_hidden_dim: 共享层隐藏单元数
        :param cls_hidden_dim: 分类头隐藏单元数
        :param reg_hidden_dim: 回归头隐藏单元数
        """
        super(MultiTaskNewsFeedModel, self).__init__()
        # 共享层: 将输入映射到一个低维公共空间
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, shared_hidden_dim),
            nn.ReLU()
        )
        # 分类任务头 (例如预测点击概率)
        self.cls_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, cls_hidden_dim),
            nn.ReLU(),
            nn.Linear(cls_hidden_dim, 1)  # 输出logit
        )
        # 回归任务头 (例如预测用户停留时间或评分)
        self.reg_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, reg_hidden_dim),
            nn.ReLU(),
            nn.Linear(reg_hidden_dim, 1)  # 输出连续值
        )
    
    def forward(self, x):
        # x: (batch_size, input_dim)
        shared_features = self.shared_fc(x)  # (batch_size, shared_hidden_dim)
        cls_logits = self.cls_head(shared_features)  # (batch_size, 1)
        reg_output = self.reg_head(shared_features)    # (batch_size, 1)
        return cls_logits, reg_output

# ------------------------------
# 示例：训练与推理
# ------------------------------
if __name__ == "__main__":
    # 假设输入特征维度为100（例如对新闻、用户和上下文特征的拼接向量）
    input_dim = 100
    shared_hidden_dim = 64
    cls_hidden_dim = 32
    reg_hidden_dim = 32
    
    model = MultiTaskNewsFeedModel(input_dim, shared_hidden_dim, cls_hidden_dim, reg_hidden_dim)
    
    # 示例数据：batch_size = 16
    batch_size = 16
    # 随机生成输入特征
    x = torch.randn(batch_size, input_dim)
    # 模拟分类任务标签：0/1（例如点击与否）
    cls_labels = torch.randint(0, 2, (batch_size, 1)).float()
    # 模拟回归任务标签：例如用户停留时间（单位：秒）
    reg_labels = torch.randn(batch_size, 1) * 10 + 30  # 均值30，方差10
    
    # 定义两个任务的损失函数
    cls_loss_fn = nn.BCEWithLogitsLoss()   # 分类任务使用二分类交叉熵（含sigmoid）
    reg_loss_fn = nn.MSELoss()             # 回归任务使用均方误差
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练简单示例
    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        cls_logits, reg_output = model(x)
        
        # 分类任务损失：将logits与二进制标签比较（内部会计算 sigmoid）
        loss_cls = cls_loss_fn(cls_logits, cls_labels)
        # 回归任务损失
        loss_reg = reg_loss_fn(reg_output, reg_labels)
        
        # 总损失可以加权求和（这里简单相加）
        loss = loss_cls + loss_reg
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, cls_loss: {loss_cls.item():.4f}, reg_loss: {loss_reg.item():.4f}, total_loss: {loss.item():.4f}")
    
    # 推理示例：预测分类概率和回归值
    model.eval()
    with torch.no_grad():
        cls_logits, reg_output = model(x)
        cls_probs = torch.sigmoid(cls_logits)
        print("Predicted click probabilities:\n", cls_probs.squeeze().cpu().numpy())
        print("Predicted regression outputs:\n", reg_output.squeeze().cpu().numpy())
        print("Labels for classification", cls_labels)
        print("Labels for regression", reg_labels)
