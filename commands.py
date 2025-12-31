import hydra
from omegaconf import DictConfig

from banana_ripeness_classification.cache_features import extract_and_cache_features
from banana_ripeness_classification.train_cached import train_cached


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.command == "cache-features":
        extract_and_cache_features(cfg.preprocessing)
    elif cfg.command == "train-cached":
        train_cached(cfg)
    else:
        print(f"Неизвестная команда: {cfg.command}")


if __name__ == "__main__":
    main()
