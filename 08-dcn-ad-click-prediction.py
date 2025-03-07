import torch
import torch.nn as nn
import torch.optim as optim

# 定义 Cross Layer
class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        # 学习参数：权重和偏置
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
    
    def forward(self, x0, x):
        # x0: 原始输入, x: 当前层输出, 两者形状均为 (batch_size, input_dim)
        # 计算内积：(w^T * x) 形状为 (batch_size, 1)
        dot = torch.sum(x * self.weight, dim=1, keepdim=True)
        # x0 * (w^T * x) 通过广播机制得到 (batch_size, input_dim)
        x0_dot = x0 * dot
        # 加上偏置和跳跃连接
        out = x0_dot + self.bias + x
        return out
    
class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.model(x)

# 定义 DCN 模型
class DCN(nn.Module):
    def __init__(self, input_dim, cross_num=2, deep_layers=[128, 64]):
        """
        :param input_dim: 输入特征维度
        :param cross_num: Cross 网络层数
        :param deep_layers: Deep 网络各层的神经元个数列表
        """
        super(DCN, self).__init__()
        # Cross 部分
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(cross_num)])
        
        # Deep 部分，构建多层感知机
        '''
        deep_input_dim = input_dim
        deep_layers_list = []
        for hidden_dim in deep_layers:
            deep_layers_list.append(nn.Linear(deep_input_dim, hidden_dim))
            deep_layers_list.append(nn.ReLU())
            deep_input_dim = hidden_dim
        self.deep = nn.Sequential(*deep_layers_list)
        '''

        deep_input_dim = input_dim
        deep_layers_list = []
        for hidden_dim in deep_layers:
            deep_layers_list.append(MLPLayer(deep_input_dim, hidden_dim))
            deep_input_dim = hidden_dim
        self.deep = nn.Sequential(*deep_layers_list)
        
        # 融合 Cross 与 Deep 输出后的全连接层
        # Cross 输出维度为 input_dim, Deep 输出维度为 deep_input_dim
        final_input_dim = input_dim + deep_input_dim
        self.final_linear = nn.Linear(final_input_dim, 1)  # 输出一个 Logit 值
    
    def forward(self, x):
        # x: 输入特征 (batch_size, input_dim)
        x0 = x
        xl = x0
        # Cross 网络：层层传递
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        cross_out = xl  # (batch_size, input_dim)
        
        # Deep 网络
        deep_out = self.deep(x)  # (batch_size, deep_output_dim)
        
        # 拼接两部分
        combined = torch.cat([cross_out, deep_out], dim=1)
        logits = self.final_linear(combined)
        return logits

# 示例：使用 DCN 进行广告点击预测
if __name__ == "__main__":
    # 假设输入特征维度为30（例如用户、广告及上下文特征的拼接向量）
    input_dim = 30
    model = DCN(input_dim=input_dim, cross_num=2, deep_layers=[128, 64])
    
    # 模拟数据：batch_size 为64
    batch_size = 64
    x = torch.randn(batch_size, input_dim)
    # 二分类标签：点击为1，未点击为0
    y = torch.randint(0, 2, (batch_size, 1)).float()
    
    # 定义损失函数和优化器，使用二分类交叉熵损失（包含 sigmoid 操作）
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 简单训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # 推理示例：计算点击概率
    model.eval()
    with torch.no_grad():
        logits = model(x)
        # Sigmoid 激活得到概率
        probs = torch.sigmoid(logits)
        print(probs.shape)
        print("预测点击概率：", probs.squeeze().cpu().numpy())
