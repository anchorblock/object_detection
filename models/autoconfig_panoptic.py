from automodel_panoptic import PANOPTIC_CONFIG_MAPPING_NAMES

def AutoPanopticConfig(model_type, *args, **kwargs):
    if model_type in PANOPTIC_CONFIG_MAPPING_NAMES:
        return PANOPTIC_CONFIG_MAPPING_NAMES[model_type](*args, **kwargs)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

