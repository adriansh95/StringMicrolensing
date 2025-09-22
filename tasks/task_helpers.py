import importlib.util
from pathlib import Path

def load_plugin_from_path(plugin_path):
    plugin_path = Path(plugin_path)
    spec = importlib.util.spec_from_file_location("plugin_module", plugin_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.plugin_func
    return result
