from .config_parser import args
from pydantic import BaseModel
from strictyaml import YAML, load
from typing import Dict, List, Optional, Sequence
import yaml
from pathlib import Path

ROOT = Path.cwd().resolve()
CONFIG_FILE_PATH = ROOT / "rubix/rubix/config.yml"


class CubeConfig(BaseModel):
    """Cube config object."""
    cube_dim: int
    num_moves_reset: int


class NetworkConfig(BaseModel):
    """Network config object."""
    ...


class ModelConfig(BaseModel):
    """Model config object."""
    ...


class Config(BaseModel):
    """Master config object."""

    cube_config: CubeConfig
    network_config: NetworkConfig
    model_config: ModelConfig


def get_config_file_path() -> Path:
    """Get the configuration file path."""
    if CONFIG_FILE_PATH.exists():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def get_config_from_yml() -> YAML:
    """Parse YAML containing the package configuration."""
    config_file_path = get_config_file_path()
    config = yaml.safe_load(open(config_file_path))
    return config


def update_config(args: Dict, config: Dict) -> Dict:
    """Update config from yml with command line arguments"""
    dic = {k: v for k, v in vars(args).items() if v is not None}
    config.update(dic)
    return config


def create_and_validate_config() -> Config:
    """Create Config item from yaml file and command line arguments"""
    config_yml = get_config_from_yml()
    parsed_config = update_config(args, config_yml)
    _config = Config(
        cube_config=CubeConfig(**parsed_config),
        network_config=NetworkConfig(**parsed_config),
        model_config=ModelConfig(**parsed_config)
    )
    return _config


config = create_and_validate_config()

if __name__ == "__main__":

    ...
