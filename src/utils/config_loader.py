
from pathlib import Path

import yaml

# src/utils/config_loader.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a YAML mapping.")

    return config


def project_path(relative_path: str) -> Path:
    path = Path(relative_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path
