import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# 生成示例数据
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用逻辑回归训练分类器
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测概率
y_probs = model.predict_proba(X_test)[:, 1]

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

youden_idx = np.argmax(tpr - fpr)
best_threshold = thresholds[youden_idx]
print(f"Best Threshold:{best_threshold:.2f}")

# 计算 AUC 值
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # 随机分类器的对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 计算PR值
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# 找到精确率和召回率均较高的阈值
f1_scores = (2 * precision * recall) / (precision + recall + 1e-8) 
best_idx = np.argmax(f1_scores) 
best_threshold = thresholds[best_idx] 
print(f"Best Threshold: {best_threshold:.2f}")

# 计算平均精确率 (Average Precision, AP)
ap_score = average_precision_score(y_test, y_probs)

# 绘制PR曲线
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR Curve (AP={ap_score:.2f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.legend(loc='lower left')
plt.grid(True)
plt.show( )
