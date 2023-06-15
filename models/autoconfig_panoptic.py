import importlib
from collections import OrderedDict
from .automodel_panoptic import PANOPTIC_CONFIG_MAPPING_NAMES


def AutoPanopticConfig(model_type, *args, **kwargs):
    """
    Creates a configuration object for AutoPanopticConfig, any panoptic model confuguration for panoptic segmentation and instance recognition.

    Args:
        model_type (str): The type of AutoPanoptic model. Available options: 'detr', 'maskformer', 'mask2former', 'custom_mask2former', 'custom_maskformer', 'oneformer'.

        *args: Additional positional arguments specific to the chosen model type.
        **kwargs: Additional keyword arguments specific to the chosen model type.

    Returns:
        config (class): A config class containing the configuration settings for panoptic model.

    Raises:
        ValueError: If an unsupported model_type is provided.

    Example:
        >>> config = AutoPanopticConfig(model_type = "custom_mask2former", backbone_config=FocalNetConfig())
    """
    module_name = 'models'  # The panoptic configs are in a module named 'models'
    
    try:
        module = importlib.import_module(f"{module_name}", "transformers")
        class_map = {key: getattr(module, value) for key, value in PANOPTIC_CONFIG_MAPPING_NAMES.items()}
        
        if model_type in class_map:
            return class_map[model_type](*args, **kwargs)
        else:
            raise ValueError(f"Invalid class name: {model_type}")
    except ImportError:
        raise ImportError(f"Module '{module_name}' not found.")

