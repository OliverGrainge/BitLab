import os
import yaml


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