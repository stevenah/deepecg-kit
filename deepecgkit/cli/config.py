"""Configuration file loading for the CLI."""

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    content = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(content)
    elif path.suffix == ".json":
        return json.loads(content)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}. Use .yaml, .yml, or .json")
