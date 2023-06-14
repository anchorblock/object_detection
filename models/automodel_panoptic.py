from collections import OrderedDict
from transformers.models.auto.auto_factory import _LazyAutoMapping, _BaseAutoModelClass, auto_class_update, model_type_to_module_name
import importlib
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES



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
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        return custom_getattribute_from_module(self._modules[module_name], attr)



MODEL_FOR_PANOPTIC_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Universal Segmentation mapping
        ("detr", "DetrForSegmentation"),
        ("mask2former", "CustomMask2FormerForUniversalSegmentation"),
        ("maskformer", "CustomMaskFormerForInstanceSegmentation"),
        ("oneformer", "OneFormerForUniversalSegmentation"),
    ]
)


MODEL_FOR_PANOPTIC_SEGMENTATION_MAPPING = _CustomLazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_PANOPTIC_SEGMENTATION_MAPPING_NAMES
)

class AutoModelForPanopticSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PANOPTIC_SEGMENTATION_MAPPING


AutoModelForPanopticSegmentation = auto_class_update(
    AutoModelForPanopticSegmentation, head_doc="panoptic segmentation"
)







# from transformers import AutoConfig, AutoModel

# AutoConfig.register("new-model", NewModelConfig)
# AutoModel.register(NewModelConfig, NewModel)

# If your NewModelConfig is a subclass of ~transformer.PretrainedConfig, make sure its model_type attribute is set to the same key you use when registering the config (here "new-model").

# Likewise, if your NewModel is a subclass of PreTrainedModel, make sure its config_class attribute is set to the same class you use when registering the model (here NewModelConfig).


