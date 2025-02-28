import torch
from transformers import AutoTokenizer, AutoModel
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
import boto3

# 使用 BGE 模型进行嵌入
class BGEEmbedder:
    def __init__(self, model_name="BAAI/bge-large-en"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embeddings = self.model(**inputs).pooler_output
        embeddings = embeddings.cpu().numpy()
        return embeddings

# 初始化 OpenSearch 客户端
def init_opensearch(host, port, index_name):
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body={
            "settings": {
                "index.knn": True  # 启用 kNN
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1024,  # 嵌入向量的维度
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    }
                }
            }
        })
    return client

# 插入文档到 OpenSearch 向量索引
def index_documents(client, index_name, docs, embedder):
    actions = []
    embeddings = embedder.embed(docs)
    for i, embedding in enumerate(embeddings):
        action = {
            "_index": index_name,
            "_id": i,
            "_source": {
                "embedding": embedding.tolist(),
                "text": docs[i]
            }
        }
        actions.append(action)
    bulk(client, actions)

# 从 OpenSearch 中检索文档
def search_documents(client, index_name, query, embedder, k=3):
    query_embedding = embedder.embed([query])[0]
    response = client.search(
        index=index_name,
        body={
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding.tolist(),
                        "k": k
                    }
                }
            }
        }
    )
    return [hit['_source']['text'] for hit in response['hits']['hits']]

# 调用 Bedrock Claude 3.5 Sonnet 进行文本生成
def generate_with_claude(query, context, bedrock_client, model_id="bedrock-runtime"):
    prompt = f"Context: {context}\nQuery: {query}\nGenerate a response based on the context."
    
    response = bedrock_client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body={"prompt": prompt, "max_tokens": 200}
    )
    return response["result"]

# 设置 AWS Bedrock 客户端
def init_bedrock():
    session = boto3.Session()
    client = session.client(service_name="bedrock-runtime")
    return client

# 示例 RAG 流程
def main():
    # AWS OpenSearch 和 Bedrock 参数
    opensearch_host = "your-opensearch-host"
    opensearch_port = 443
    index_name = "your-index"
    
    # 示例文档和查询
    docs = [
        "Deep learning is a subset of machine learning...",
        "OpenSearch is a distributed search and analytics engine...",
        "Transformers are a type of neural network architecture..."
    ]
    query = "What is OpenSearch?"
    
    # 初始化嵌入模型和 OpenSearch
    embedder = BGEEmbedder()
    client = init_opensearch(opensearch_host, opensearch_port, index_name)
    
    # 插入文档到 OpenSearch 向量索引
    index_documents(client, index_name, docs, embedder)
    
    # 从 OpenSearch 检索相关文档
    retrieved_docs = search_documents(client, index_name, query, embedder)
    context = " ".join(retrieved_docs)
    
    # 初始化 Bedrock Claude 3.5 客户端
    bedrock_client = init_bedrock()
    
    # 调用 Claude 3.5 Sonnet 生成答案
    response = generate_with_claude(query, context, bedrock_client)
    print(response)

if __name__ == "__main__":
    main()
