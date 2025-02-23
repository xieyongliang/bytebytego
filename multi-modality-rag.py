# 安装依赖库
# !pip install opensearch-py torch torchvision transformers faiss-cpu opencv-python

import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
from torchvision import transforms
import faiss
import numpy as np
import cv2
from opensearchpy import OpenSearch

# 1. 文本处理和嵌入生成（使用bge模型）
class TextEmbedder:
    def __init__(self, model_name='BAAI/bge-large-en'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, text):
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            embeddings = self.model(**tokens).last_hidden_state.mean(dim=1)  # 获取均值作为文本的嵌入
        return embeddings.cpu().numpy()

# 2. 图片处理和嵌入生成（使用CLIP模型）
class ImageEmbedder:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.model = AutoModel.from_pretrained(model_name)
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def embed(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            image_embedding = self.model.get_image_features(image_tensor)
        return image_embedding.cpu().numpy()

# 3. 视频处理和嵌入生成（通过抽取关键帧并使用CLIP）
class VideoEmbedder:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.image_embedder = ImageEmbedder(model_name=model_name)

    def extract_keyframes(self, video_path, num_frames=5):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(frame)
        cap.release()
        return frames

    def embed(self, video_path):
        frames = self.extract_keyframes(video_path)
        embeddings = [self.image_embedder.embed(frame) for frame in frames]
        return np.mean(embeddings, axis=0)

# 4. 向量存储和索引（使用 AWS OpenSearch）
class VectorIndexer:
    def __init__(self, index_name='multimodal-index'):
        self.client = OpenSearch(
            hosts=[{'host': 'your-opensearch-endpoint', 'port': 443}],
            http_auth=('username', 'password'),
            use_ssl=True,
            verify_certs=True,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )
        self.index_name = index_name
        self.create_index()

    def create_index(self):
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param": {
                        "m": 24,  # HNSW参数
                        "ef_construction": 200
                    }
                }
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": 768  # 嵌入向量的维度
                    },
                    "content": {
                        "type": "text"
                    },
                    "modality": {
                        "type": "keyword"
                    }
                }
            }
        }
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=index_body)

    def index_document(self, embedding, content, modality):
        doc = {
            "vector": embedding.tolist(),
            "content": content,
            "modality": modality
        }
        self.client.index(index=self.index_name, body=doc)

    def search(self, query_vector, k=5):
        search_body = {
            "size": k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_vector.tolist(),
                        "k": k
                    }
                }
            }
        }
        return self.client.search(index=self.index_name, body=search_body)

# 5. 综合多模态嵌入和索引操作
if __name__ == '__main__':
    # 初始化嵌入生成器
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder()
    video_embedder = VideoEmbedder()

    # 初始化向量索引器
    indexer = VectorIndexer()

    # 文本嵌入和索引
    text = "This is a sample document about AI."
    text_embedding = text_embedder.embed(text)
    indexer.index_document(text_embedding[0], content=text, modality="text")

    # 图片嵌入和索引
    image_path = 'path_to_image.jpg'
    image_embedding = image_embedder.embed(image_path)
    indexer.index_document(image_embedding[0], content="Sample Image", modality="image")

    # 视频嵌入和索引
    video_path = 'path_to_video.mp4'
    video_embedding = video_embedder.embed(video_path)
    indexer.index_document(video_embedding[0], content="Sample Video", modality="video")

    # 搜索示例
    query = "Artificial intelligence"
    query_embedding = text_embedder.embed(query)[0]
    results = indexer.search(query_embedding)

    for result in results['hits']['hits']:
        print(f"Found document: {result['_source']['content']} (Modality: {result['_source']['modality']})")
