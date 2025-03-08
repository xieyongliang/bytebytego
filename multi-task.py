from torch.utils.data import Dataset, DataLoader
import torch
import clip
import cv2
from PIL import Image
import os
import numpy as np

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to extract frames from video
def extract_frames(video_path, frame_rate=1):
    video = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if count % frame_rate == 0:  # Extract 1 frame per second
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB
        count += 1
    video.release()
    return frames

# Function to get image embeddings from CLIP
def get_image_embeddings(frames):
    embeddings = []
    for frame in frames:
        image = Image.fromarray(frame)
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            embeddings.append(image_features)
    return torch.cat(embeddings).mean(dim=0)

class ContrastiveLearningDataset(Dataset):
    def __init__(self, emb_model, df):
        self.df = df
        self.emb_model = emb_model

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_or_video_path = self.df.iloc[index].image_or_videos
        image_emb, video_emb = torch.nan, torch.nan
        _, file_extension = os.path.splitext(image_or_video_path)
        if file_extension in ['jpg', 'jpeg', 'png']:
            image = Image.open(image_or_video_path)
            image = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_emb = model.encode_image(image).squeeze(dim=0)
        else:
            frames = extract_frames(image_or_video_path, frame_rate=1)
            video_emb = get_image_embeddings(frames)

        txt_emb = self.emb_model(self.df.iloc[index].texual_content)
        return torch.concat([txt_emb, image_emb, video_emb])
    
if __name__ == '__main__':
    '''    
    image_or_video_path = '/Users/yonglxie/Downloads/arab.mp4'
    frames = extract_frames(image_or_video_path, frame_rate=1)
    video_emb = get_image_embeddings(frames)
    print(video_emb.shape)
    image_emb = torch.zeros(512)
    print(image_emb.shape)
    '''
    image_or_video_path = '/Users/yonglxie/Downloads/image-1.jpg'
    image = Image.open(image_or_video_path)
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_emb = model.encode_image(image).squeeze(dim=0)
        print(image_emb.shape)
  