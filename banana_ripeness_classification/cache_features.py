import subprocess
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel


def extract_and_cache_features(cfg: DictConfig):
    """
    Извлекает признаки CLIP из изображений и сохраняет их.
    cfg: Конфигурация Hydra (preprocessing)
    """

    data_path = Path(cfg.data_dir)
    cache_path = Path(cfg.cache_dir)
    batch_size = cfg.batch_size
    model_name = cfg.model_name
    device = cfg.device

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    subprocess.run(["dvc", "pull"])
    cache_path.mkdir(exist_ok=True)

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPVisionModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    def transform(image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    for split in ["train", "valid"]:
        split_path = data_path / split
        if not split_path.exists():
            continue

        dataset = ImageFolder(split_path, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"Извлечение признаков ({split})"):
                images = images.to(device)
                outputs = model(pixel_values=images)
                features = outputs.pooler_output.cpu()  # [batch, hidden_size]

                all_features.append(features)
                all_labels.append(labels)

        features_tensor = torch.cat(all_features, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        torch.save(
            {
                "features": features_tensor,
                "labels": labels_tensor,
                "classes": dataset.classes,
            },
            cache_path / f"{split}_features.pt",
        )


if __name__ == "__main__":
    extract_and_cache_features()
