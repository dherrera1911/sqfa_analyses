import numpy as np
import torch

def load_data(data_type: str):
    # Validate input
    if data_type not in ["train", "test"]:
        raise ValueError("data_type must be either 'train' or 'test'.")

    videos_path = f"data/videos_{data_type}_int.csv"
    labels_path = f"data/labels_{data_type}.csv"
    category_values_path = "data/category_values.csv"

    # Load data using numpy
    videos_np = np.loadtxt(videos_path, delimiter=",", dtype=np.float32)
    labels_np = np.loadtxt(labels_path, delimiter=",", dtype=np.int64)
    category_values_np = np.loadtxt(category_values_path, delimiter=",", dtype=np.float32)

    # Convert to torch tensors
    videos = torch.from_numpy(videos_np)
    labels = torch.from_numpy(labels_np)
    category_values = torch.from_numpy(category_values_np)

    # Undo the transformation for compressing videos for NEURIPS submission
    videos = videos / 10000
    videos= videos - videos.mean(dim=1, keepdim=True)

    return videos, labels, category_values


def normalize_stim(videos, c50):
    """Apply divisive normalization by norm + c50"""
    return videos / torch.sqrt(torch.norm(videos, dim=1, keepdim=True)**2 + c50)

