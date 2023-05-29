from transformers import FocalNetForImageClassification, FocalNetConfig, FocalNetModel
import torch
import json


# read basic configurations
with open("dev/focalconfig.json", 'r') as json_file:
    # Load the JSON data
    config = json.load(json_file)


# read id2label and label2id
with open("configs/datasets/imagenet-1k-id2label.json", 'r') as json_file:
    # Load the JSON data
    id2label = json.load(json_file)


with open("configs/datasets/imagenet-1k-label2id.json", 'r') as json_file:
    # Load the JSON data
    label2id = json.load(json_file)

config["id2label"] = id2label
config["label2id"] = label2id

# print(config)
configuration = FocalNetConfig(**config)
model = FocalNetForImageClassification(configuration)

# print(model)

extractor = FocalNetModel(configuration)

# print(extractor)

# model.config.id2label = {0: "label0", 1:"label1", 2:"label2"}
# print(model.config)
# print(model)
