import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import nn, optim
import torchvision.transforms as T
from PIL import Image

# 加载预训练的CLIP模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 使用自定义数据集（这里以COCO数据集为例，其他数据集类似）
dataset = load_dataset("coco", split="train[:10%]")  # 加载部分数据

# 定义数据预处理：包括对图像的缩放和标准化
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

# 定义对比损失函数
class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, text_features, image_features):
        # 计算对比损失：使用余弦相似度
        logits_per_image = torch.matmul(image_features, text_features.T)
        logits_per_text = logits_per_image.T
        labels = torch.arange(image_features.size(0)).to(logits_per_image.device)
        
        # 图像-文本对之间的损失
        loss_img_text = nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_text_img = nn.CrossEntropyLoss()(logits_per_text, labels)
        loss = (loss_img_text + loss_text_img) / 2
        return loss

# 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-6)

# 数据加载器
def collate_fn(batch):
    images = [transform(Image.open(item['image']['coco_url']).convert("RGB")) for item in batch]
    texts = [item['caption'] for item in batch]
    return {
        'images': torch.stack(images),
        'texts': texts
    }

dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

# 微调循环
model.train()
for epoch in range(3):  # 训练3个epoch
    total_loss = 0
    for batch in dataloader:
        images = batch['images'].to(device)
        texts = batch['texts']

        # 处理文本和图像输入
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
        
        # 获取模型的输出
        image_features = model.get_image_features(**inputs)
        text_features = model.get_text_features(**inputs)

        # 计算损失
        loss = CLIPLoss()(text_features, image_features)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

# 保存微调后的模型
model.save_pretrained("./finetuned_clip")
processor.save_pretrained("./finetuned_clip")
