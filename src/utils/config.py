from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
import os

class Config(dict):
    """Configuration class that supports both dictionary and dot notation access."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._convert_to_config_recursive(self)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    @staticmethod
    def _convert_to_config_recursive(obj):
        """Recursively convert nested dictionaries to Config objects."""
        for key, value in obj.items():
            if isinstance(value, dict):
                obj[key] = Config(value)
                Config._convert_to_config_recursive(obj[key])
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

# Default configuration
config = Config({
    "model": {
        "teacher": {
            "name": "deepseek-reasoner",
            "device": "cuda"
        },
        "student": "gpt2",
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.5
    },
    "training": {
        "batch_size": 8,
        "num_epochs": 3,
        "learning_rate": 5e-5,
        "max_length": 128
    },
    "data": {
        "dataset": "wikitext",
        "split": "train"
    }
})
