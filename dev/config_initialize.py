######### Architectures ######### 

## DETR

import torch
from transformers import DetrFeatureExtractor, DetrConfig

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
config = DetrConfig.from_pretrained("facebook/detr-resnet-50-panoptic")

print(config)




## Maskformer

## mask2former

## oneformer


## DINO

