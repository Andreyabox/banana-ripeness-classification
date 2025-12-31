import subprocess
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from pathlib import Path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from banana_ripeness_classification.cached_dataset import CachedFeaturesDataset
from banana_ripeness_classification.model_cached import CachedCLIPClassifier


def get_git_commit_id() -> str:
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return result.decode().strip()
    except Exception:
        return "unknown"


def train_cached(cfg: DictConfig):
    """
        cfg: Полная конфигурация Hydra
    """
    cache_path = Path(cfg.preprocessing.cache_dir)
    batch_size = cfg.train.batch_size
    epochs = cfg.train.epochs
    lr = cfg.model.lr
    num_workers = cfg.train.num_workers

    train_dataset = CachedFeaturesDataset(cache_path / "train_features.pt")
    val_dataset = CachedFeaturesDataset(cache_path / "valid_features.pt")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = CachedCLIPClassifier(
        hidden_size=train_dataset.features.shape[1],  # 768 для CLIP ViT-B/32
        num_classes=len(train_dataset.classes),
        lr=lr,
    )

    git_commit_id = get_git_commit_id()

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        log_model=cfg.mlflow.log_model,
    )

    hyperparams = {
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "num_workers": num_workers,
        "hidden_size": train_dataset.features.shape[1],
        "num_classes": len(train_dataset.classes),
        "model_name": cfg.model.name,
        "preprocessing_model": cfg.preprocessing.model_name,
        "git_commit_id": git_commit_id,
    }
    mlflow_logger.log_hyperparams(hyperparams)

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=cfg.train.log_every_n_steps,
        logger=mlflow_logger,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return trainer, model


if __name__ == "__main__":
    train_cached()
