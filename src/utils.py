import os
import yaml
from pathlib import Path


def load_env():
    """
    Load environment variables from .env file if it exists.
    Call this function at the very top of executable scripts (before other imports)
    to ensure environment variables are available when packages are imported.
    """
    try:
        from dotenv import load_dotenv
        # Look for .env file in the repository root (parent of src/)
        repo_root = Path(__file__).parent.parent
        env_path = repo_root / '.env'
        print(f"Loading environment variables from {env_path}")
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError as e:
        raise ImportError(
            "python-dotenv is required for loading the .env file. "
            "Please install it with 'pip install python-dotenv'."
        ) from e


def get_data_dir() -> str:
    """
    Root for all BitLab data not stored in the HuggingFace cache (e.g. partial
    dataset downloads). Set BITLAB_DATA_DIR to override; default is "data".
    output_dir and data_path are relative to this root.
    """
    return os.environ.get("BITLAB_DATA_DIR", "data")


def data_path(relative: str) -> str:
    """Return absolute path for a relative path under the data root."""
    return os.path.join(get_data_dir(), relative)


def load_config(yaml_path: str) -> dict:
    """
    Loads a YAML configuration file and returns its contents as a dictionary.
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config