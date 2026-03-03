"""Configuration file loading for the CLI."""

import json
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    content = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "PyYAML is required for YAML config files. Install it with: pip install pyyaml"
            ) from e
        return yaml.safe_load(content)
    elif path.suffix == ".json":
        return json.loads(content)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}. Use .yaml, .yml, or .json")
