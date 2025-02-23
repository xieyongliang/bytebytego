import openai
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# 配置OpenAI API
openai.api_key = 'your-openai-api-key'

# 加载bge embedding模型
model_name = "BAAI/bge-base-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model.to(device)

# 函数：从文本生成嵌入向量
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).pooler_output
    return embeddings.cpu().numpy()

# 构建FAISS索引
dimension = 768  # 假设 bge embedding 输出的向量维度为 768
index = faiss.IndexFlatL2(dimension)  # 使用L2距离

# 示例文档库 (可以根据需求替换成真实文档)
documents = [
    "The Eiffel Tower is in Paris.",
    "Python is a popular programming language.",
    "The human brain consists of billions of neurons.",
    "Machine learning is a subset of artificial intelligence."
]

# 生成文档嵌入并添加到FAISS索引
document_embeddings = np.vstack([generate_embedding(doc) for doc in documents])
index.add(document_embeddings)

# 查询阶段：从用户输入文本生成嵌入并查找相关文档
def retrieve_documents(query, k=2):
    query_embedding = generate_embedding(query)
    distances, indices = index.search(query_embedding, k)  # 返回最近的k个文档
    return [(documents[i], distances[0][j]) for j, i in enumerate(indices[0])]

# 生成阶段：使用ChatGPT生成回答
def generate_answer(query, context):
    prompt = f"Given the following context: {context}\n\nAnswer the following question: {query}"
    
    response = openai.Completion.create(
        engine="text-davinci-003",  # 替换为ChatGPT模型，如gpt-3.5-turbo
        prompt=prompt,
        max_tokens=150
    )
    
    return response.choices[0].text.strip()

# 主流程：RAG模型
def rag_pipeline(query):
    # Step 1: 检索相关文档
    retrieved_docs = retrieve_documents(query)
    context = " ".join([doc for doc, _ in retrieved_docs])
    
    # Step 2: 使用ChatGPT生成回答
    generated_answer = generate_answer(query, context)
    
    return generated_answer

# 示例查询
query = "What is the Eiffel Tower and where is it located?"
answer = rag_pipeline(query)

print("Query:", query)
print("Answer:", answer)
