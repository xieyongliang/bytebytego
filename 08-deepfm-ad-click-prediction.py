import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim, deep_layers=[128, 64], dropout=[0.5, 0.5]):
        """
        :param field_dims: 每个字段（特征）的取值个数列表，例如 [100, 200, 50, ...]
        :param embed_dim: 嵌入向量的维度
        :param deep_layers: 深度网络各层神经元数
        :param dropout: 每层dropout比例（可选）
        """
        super(DeepFM, self).__init__()
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim

        # 1. Embedding layer：每个字段都有自己的嵌入矩阵
        # nn.ModuleList存放每个字段的嵌入层
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in field_dims])
        # 线性部分: 对每个字段使用单独的embedding来表示一阶特征权重
        self.linear_embeddings = nn.ModuleList([nn.Embedding(dim, 1) for dim in field_dims])
        self.linear_bias = nn.Parameter(torch.zeros(1))

        # 2. Deep 部分：构建多层感知机
        deep_input_dim = self.num_fields * embed_dim
        deep_layers_list = []
        for i, layer_size in enumerate(deep_layers):
            deep_layers_list.append(nn.Linear(deep_input_dim, layer_size))
            deep_layers_list.append(nn.ReLU())
            deep_layers_list.append(nn.Dropout(dropout[i]))
            deep_input_dim = layer_size
        self.deep = nn.Sequential(*deep_layers_list)

        # 输出层，将深度网络输出与FM输出拼接后映射到一个logit
        # FM二阶输出为 embed_dim 维（标量由平方差公式得出）
        # 线性部分输出为标量
        # 深度网络输出为 deep_layers[-1] 维，经过全连接层映射为标量
        self.deep_out = nn.Linear(deep_input_dim, 1)
        self.final_linear = nn.Linear(2, 1)  # 2个标量：FM输出和深度网络输出（线性部分已经包含在FM部分中）

    def forward(self, x):
        """
        :param x: 输入张量，形状为 (batch_size, num_fields)，每个元素是特征索引
        """
        batch_size = x.size(0)
        
        # 1. 线性部分：对每个字段查找对应的一阶权重并求和
        linear_terms = [emb(x[:, i]) for i, emb in enumerate(self.linear_embeddings)]
        # 每个 linear_terms[i] 的形状为 (batch_size, 1)
        linear_part = torch.sum(torch.cat(linear_terms, dim=1), dim=1, keepdim=True) + self.linear_bias  # (batch_size, 1)

        # 2. Embedding lookup for each field (for FM and deep part)
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        # embed_list[i] shape: (batch_size, embed_dim)
        # Stack embeddings: shape (batch_size, num_fields, embed_dim)
        x_embed = torch.stack(embed_list, dim=1)
        
        # 3. FM 部分：计算二阶交互
        # 公式：0.5 * ( (sum_i v_i)^2 - sum_i v_i^2 )
        sum_of_embeddings = torch.sum(x_embed, dim=1)   # (batch_size, embed_dim)
        sum_square = sum_of_embeddings * sum_of_embeddings  # element-wise square, (batch_size, embed_dim)
        square_sum = torch.sum(x_embed * x_embed, dim=1)  # (batch_size, embed_dim)
        fm_part = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)  # (batch_size, 1)

        # 4. Deep 部分：将所有字段的嵌入拼接后输入MLP
        deep_input = x_embed.view(batch_size, -1)  # (batch_size, num_fields * embed_dim)
        deep_out = self.deep(deep_input)           # (batch_size, last_deep_dim)
        deep_out = self.deep_out(deep_out)         # (batch_size, 1)
        
        # 5. 输出层：将 FM 部分和 Deep 部分结果相加，再加上线性部分
        # 此处通常的组合方式有很多种，可采用直接相加或者再经过一个全连接层融合
        total_sum = linear_part + fm_part + deep_out
        # 预测概率
        y_pred = torch.sigmoid(total_sum) #(batch_size, 1)
        return y_pred

# ---------------------------
# 示例：使用DeepFM进行广告点击预测
# ---------------------------
if __name__ == "__main__":
    # 假设有5个字段，每个字段的取值个数如下（例如：用户ID、广告ID、时间、位置、设备）
    field_dims = [1000, 500, 24, 100, 10]  # 可根据实际情况设置
    embed_dim = 8  # 嵌入向量维度
    model = DeepFM(field_dims, embed_dim, deep_layers=[32, 16], dropout=[0.5, 0.5])
    
    # 模拟一个批次数据，batch_size = 64
    batch_size = 64
    # 每个样本是5个字段的索引（值在0~对应field_dims-1之间）
    x = torch.stack([
        torch.randint(0, field_dims[i], (batch_size,)) for i in range(len(field_dims))
    ], dim=1)  # shape: (64, 5)
    
    # 模拟标签：点击为1，未点击为0
    y = torch.randint(0, 2, (batch_size, 1)).float()

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 简单训练循环（仅示例，未划分训练/验证集）
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
