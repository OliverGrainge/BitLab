import yaml 

def load_config(yaml_path: str) -> dict:
    """
    Loads a YAML configuration file and returns its contents as a dictionary.
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config