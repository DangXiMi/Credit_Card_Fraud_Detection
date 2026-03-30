import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config():
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

SETTINGS = load_config()