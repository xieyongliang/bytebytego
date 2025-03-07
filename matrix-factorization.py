import numpy as np

class MatrixFactorization:
    def __init__(self, R, K, alpha, beta, iterations):
        """
        R: 用户-项目评分矩阵，形状 (num_users, num_items)
        K: 潜在因子数量
        alpha: 学习率
        beta: 正则化参数
        iterations: 最大迭代次数
        """
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # 随机初始化用户和项目的潜在因子矩阵
        self.P = np.random.rand(self.num_users, self.K)
        self.Q = np.random.rand(self.num_items, self.K)

        # 迭代优化
        for it in range(self.iterations):
            for i in range(self.num_users):
                for j in range(self.num_items):
                    # 仅更新已有评分的样本
                    if self.R[i, j] > 0:
                        # 计算误差：实际评分与预测评分之差
                        error_ij = self.R[i, j] - np.dot(self.P[i, :], self.Q[j, :].T)
                        # 按梯度下降更新用户和项目的潜在因子
                        for k in range(self.K):
                            self.P[i, k] += self.alpha * (2 * error_ij * self.Q[j, k] - self.beta * self.P[i, k])
                            self.Q[j, k] += self.alpha * (2 * error_ij * self.P[i, k] - self.beta * self.Q[j, k])
            
            # 计算总误差（可选，用于监控训练过程）
            error = 0
            for i in range(self.num_users):
                for j in range(self.num_items):
                    if self.R[i, j] > 0:
                        error += (self.R[i, j] - np.dot(self.P[i, :], self.Q[j, :].T)) ** 2
                        error += (self.beta/2) * (np.linalg.norm(self.P[i, :])**2 + np.linalg.norm(self.Q[j, :])**2)
            if (it+1) % 1000 == 0 or it == 0:
                print(f"Iteration: {it+1} ; error = {error:.4f}")
            # 如果误差收敛，则退出
            if error < 0.001:
                break

    def predict(self):
        # 预测评分矩阵
        return np.dot(self.P, self.Q.T)

# ------------------------------
# 示例使用
# ------------------------------
if __name__ == "__main__":
    # 示例评分矩阵 R，其中 0 表示缺失评分
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ], dtype=np.float32)

    # 设置潜在因子数量、学习率、正则化系数和迭代次数
    K = 2
    alpha = 0.002
    beta = 0.02
    iterations = 5000

    mf = MatrixFactorization(R, K, alpha, beta, iterations)
    mf.train()
    predicted_R = mf.predict()
    
    print("原始评分矩阵 R:")
    print(R)
    print("\n预测评分矩阵:")
    print(np.round(predicted_R, 2))
