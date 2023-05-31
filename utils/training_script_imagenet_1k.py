



from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
import json


config = AutoConfig.from_json_file('./config.json')

# read id2label and label2id
with open("configs/datasets/imagenet-1k-id2label.json", 'r') as json_file:
    # Load the JSON data
    id2label = json.load(json_file)


with open("configs/datasets/imagenet-1k-label2id.json", 'r') as json_file:
    # Load the JSON data
    label2id = json.load(json_file)

config["id2label"] = id2label
config["label2id"] = label2id



model = AutoModelForImageClassification(config)

image_processor = AutoImageProcessor.from_json_file('./config.json')










id2label


label2id