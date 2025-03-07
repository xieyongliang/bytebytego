import numpy as np

def extract_media_features(post):
    """
    从post中提取媒体特征，如果存在视频或图片，返回其特征向量；
    如果不存在，则返回默认向量（例如全零向量）。
    
    假设媒体特征的维度为 D
    """
    D = 128  # 媒体特征向量维度
    if post.get("media"):
        # 假设存在一个函数 extract_features 来提取视频或图片的特征
        media_features = extract_features(post["media"])
        media_flag = 1  # 标记存在媒体
    else:
        media_features = np.zeros(D)
        media_flag = 0  # 标记无媒体
    
    # 你还可以选择拼接媒体缺失标识
    # 例如返回 (media_features, media_flag)
    return media_features, media_flag

# 示例：构造一个post
post_with_media = {"id": 1, "title": "科技新闻", "media": "video_file.mp4"}
post_without_media = {"id": 2, "title": "经济新闻", "media": None}

features1, flag1 = extract_media_features(post_with_media)
features2, flag2 = extract_media_features(post_without_media)

print("Post 1 media flag:", flag1, "Features:", features1[:5])
print("Post 2 media flag:", flag2, "Features:", features2[:5])
