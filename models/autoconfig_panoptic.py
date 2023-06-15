import importlib
from collections import OrderedDict
from .automodel_panoptic import PANOPTIC_CONFIG_MAPPING_NAMES


def AutoPanopticConfig(model_type, *args, **kwargs):
    module_name = 'models'  # Assuming the models are in a module named 'models'
    
    try:
        module = importlib.import_module(f"{module_name}", "transformers")
        class_map = {key: getattr(module, value) for key, value in PANOPTIC_CONFIG_MAPPING_NAMES.items()}
        
        if model_type in class_map:
            return class_map[model_type](*args, **kwargs)
        else:
            raise ValueError(f"Invalid class name: {model_type}")
    except ImportError:
        raise ImportError(f"Module '{module_name}' not found.")

