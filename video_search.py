import os
import torch
import clip
import cv2
from PIL import Image

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
    return torch.cat(embeddings)

# Function to search videos based on text query
def search_videos(query, video_paths, frame_rate=1):
    # Get text embedding
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text)

    # Iterate over each video
    results = []
    for video_path in video_paths:
        frames = extract_frames(video_path, frame_rate)
        video_embeddings = get_image_embeddings(frames)
        
        # Compute cosine similarity between text and video frames
        similarities = torch.nn.functional.cosine_similarity(text_embedding, video_embeddings)
        max_similarity = similarities.max().item()

        # Store the max similarity score and the corresponding video path
        results.append((max_similarity, video_path))
    
    # Sort videos by similarity
    results.sort(reverse=True, key=lambda x: x[0])
    return results

# Example usage:
video_dir = "/Users/yonglxie/Downloads"
video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]

# Text query to search
query = "ConnectNow"

# Perform search
search_results = search_videos(query, video_files)

# Output the sorted video files based on similarity
for similarity, video_path in search_results:
    print(f"Video: {video_path}, Similarity: {similarity}")
