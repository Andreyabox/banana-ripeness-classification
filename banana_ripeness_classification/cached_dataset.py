import torch
from torch.utils.data import Dataset


class CachedFeaturesDataset(Dataset):
    """Dataset, который загружает предварительно извлечённые CLIP-признаки."""
    
    def __init__(self, cache_file: str):
        """
            cache_file: Путь к файлу с кэшированными признаками
        """
        data = torch.load(cache_file, weights_only=True)
        self.features = data["features"]
        self.labels = data["labels"]
        self.classes = data["classes"]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
