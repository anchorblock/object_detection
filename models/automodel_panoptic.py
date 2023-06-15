from collections import OrderedDict
from transformers.models.auto.auto_factory import _LazyAutoMapping, _BaseAutoModelClass, auto_class_update, model_type_to_module_name
import importlib


def custom_getattribute_from_module(module, attr):
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple(custom_getattribute_from_module(module, a) for a in attr)
    if hasattr(module, attr):
        return getattr(module, attr)
    # Some of the mappings have entries model_type -> object of another model type. In that case we try to grab the
    # object at the top level.
    transformers_module = importlib.import_module("models", "transformers")

    if module != transformers_module:
        try:
            return custom_getattribute_from_module(transformers_module, attr)
        except ValueError:
            raise ValueError(f"Could not find {attr} neither in {module} nor in {transformers_module}!")
    else:
        raise ValueError(f"Could not find {attr} in {transformers_module}!")
    


class _CustomLazyAutoMapping(_LazyAutoMapping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "models")
        return custom_getattribute_from_module(self._modules[module_name], attr)


PANOPTIC_CONFIG_MAPPING_NAMES = OrderedDict(
    [
        # Model for Panoptic Segmentation mapping
        ("detr", "DetrConfig"),
        ("maskformer", "MaskFormerConfig"),
        ("mask2former", "Mask2FormerConfig"),
        ("custom_mask2former", "CustomMask2FormerConfig"),
        ("custom_maskformer", "CustomMaskFormerConfig"),
        ("oneformer", "OneFormerConfig"),
    ]
)


MODEL_FOR_PANOPTIC_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Panoptic Segmentation mapping
        ("detr", "ModifiedDetrForSegmentation"),
        ("maskformer", "ModifiedMaskFormerForInstanceSegmentation"),
        ("mask2former", "ModifiedMask2FormerForUniversalSegmentation"),
        ("custom_mask2former", "CustomMask2FormerForUniversalSegmentation"),
        ("custom_maskformer", "CustomMaskFormerForInstanceSegmentation"),
        ("oneformer", "ModifiedOneFormerForUniversalSegmentation"),
    ]
)

MODEL_FOR_PANOPTIC_SEGMENTATION_MAPPING = _CustomLazyAutoMapping(
    PANOPTIC_CONFIG_MAPPING_NAMES, MODEL_FOR_PANOPTIC_SEGMENTATION_MAPPING_NAMES
)

class AutoModelForPanopticSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PANOPTIC_SEGMENTATION_MAPPING




AutoModelForPanopticSegmentation = auto_class_update(
    AutoModelForPanopticSegmentation, head_doc="panoptic segmentation"
)


