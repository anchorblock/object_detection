from collections import OrderedDict
from transformers.models.auto.auto_factory import _LazyAutoMapping, _BaseAutoModelClass, auto_class_update
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES


MODEL_FOR_PANOPTIC_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Universal Segmentation mapping
        ("detr", "DetrForSegmentation"),
        ("mask2former", "CustomMask2FormerForUniversalSegmentation"),
        ("maskformer", "MaskFormerForInstanceSegmentation"),
        ("oneformer", "OneFormerForUniversalSegmentation"),
    ]
)


MODEL_FOR_PANOPTIC_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_PANOPTIC_SEGMENTATION_MAPPING_NAMES
)

class AutoModelForPanopticSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PANOPTIC_SEGMENTATION_MAPPING


AutoModelForPanopticSegmentation = auto_class_update(
    AutoModelForPanopticSegmentation, head_doc="panoptic segmentation"
)