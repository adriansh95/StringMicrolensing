import yaml
import importlib.util
from pathlib import Path

def load_yaml_config(file_path):
    """Load parameters from a YAML file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def load_python_config(file_path):
    """Load parameters from a Python file as a dictionary."""
    spec = importlib.util.spec_from_file_location("config", file_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    result = {key: getattr(config, key) for key in 
              dir(config) if not key.startswith("__")}
    return result

def load_config(yaml_path=None, py_path=None):
    """Load configurations from both YAML and Python files."""
    config = {}
    if yaml_path:
        config.update(load_yaml_config(yaml_path))
    if py_path:
        config.update(load_python_config(py_path))
    return config
