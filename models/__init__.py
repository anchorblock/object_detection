from .detr import DetrConfig, ModifiedDetrForSegmentation
from .mask2former import Mask2FormerConfig, ModifiedMask2FormerForUniversalSegmentation
from .maskformer import MaskFormerConfig, ModifiedMaskFormerForInstanceSegmentation
from .custom_mask2former import CustomMask2FormerConfig, CustomMask2FormerForUniversalSegmentation
from .custom_maskformer import CustomMaskFormerConfig, CustomMaskFormerForInstanceSegmentation
from .oneformer import OneFormerConfig, ModifiedOneFormerForUniversalSegmentation
from .automodel_panoptic import AutoModelForPanopticSegmentation
from .autoconfig_panoptic import AutoPanopticConfig