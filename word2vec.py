from gensim.models import Word2Vec

# 示例语料，每个句子是一个词列表（通常需要大规模语料）
sentences = [
    ["我", "爱", "自然语言处理"],
    ["我", "爱", "机器学习"],
    ["自然语言处理", "是", "人工智能", "的", "一个", "分支"],
    ["机器学习", "是", "人工智能", "的", "一个", "分支"],
    ["深度学习", "是", "机器学习", "的", "一个", "分支"],
    ["人工智能", "改变", "世界"],
    ["机器学习", "使", "计算机", "更", "智能"],
]

# 训练 Word2Vec 模型
# vector_size: 每个词向量的维度
# window: 上下文窗口大小
# min_count: 忽略总频率低于min_count的词
# workers: 并行训练的线程数
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 保存模型
model.save("word2vec.model")

# 加载模型（可选）
model = Word2Vec.load("word2vec.model")

# 获取某个词的词向量，例如“机器学习”
vector = model.wv["机器学习"]
print("‘机器学习’的词向量：\n", vector)

# 查找与“机器学习”最相似的词，返回 top 3 个
similar_words = model.wv.most_similar("机器学习", topn=3)
print("与‘机器学习’最相似的词：\n", similar_words)
