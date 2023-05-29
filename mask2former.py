from transformers import MaskFormerForInstanceSegmentation, MaskFormerConfig, MaskFormerModel
import torch
import json


# read basic configurations
with open("configs/architectures/maskformer/maskformer_config.json", 'r') as json_file:
    # Load the JSON data
    maskformer_config = json.load(json_file)

# print(config)
configuration = MaskFormerConfig(**maskformer_config)
model = MaskFormerForInstanceSegmentation(configuration)

print(model)

print("********************************")


from focalnet import extractor

model.model.pixel_level_module.encoder.model = extractor

print(model)

from transformers import AutoImageProcessor
from PIL import Image
import requests


image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")


url = (

    "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"

)

image = Image.open(requests.get(url, stream=True).raw)

inputs = image_processor(images=image, return_tensors="pt")


outputs = model(**inputs)

# model predicts class_queries_logits of shape `(batch_size, num_queries)`

# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`

class_queries_logits = outputs.class_queries_logits

masks_queries_logits = outputs.masks_queries_logits

# you can pass them to image_processor for postprocessing

predicted_semantic_map = image_processor.post_process_semantic_segmentation(

    outputs, target_sizes=[image.size[::-1]]

)[0]

# we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)

print(list(predicted_semantic_map.shape))


























# # read backbone_configurations
# from v1 import config as backbone_config

# with open("decoder_config.json", 'r') as json_file:
#     # Load the JSON data
#     decoder_config = json.load(json_file)


# # read id2label and label2id
# with open("configs/datasets/imagenet-1k-id2label.json", 'r') as json_file:
#     # Load the JSON data
#     id2label = json.load(json_file)


# with open("configs/datasets/imagenet-1k-label2id.json", 'r') as json_file:
#     # Load the JSON data
#     label2id = json.load(json_file)


# maskformer_config["backbone_config"] = backbone_config
# maskformer_config["decoder_config"] = decoder_config
# maskformer_config["id2label"] = id2label
# maskformer_config["label2id"] = label2id



# # print(config)
# configuration = MaskFormerConfig(**maskformer_config)
# model = MaskFormerForInstanceSegmentation(configuration)

# print(model)

# extractor = MaskFormerModel(configuration)

# print(extractor)

# # model.config.id2label = {0: "label0", 1:"label1", 2:"label2"}
# # print(model.config)
# # print(model)
