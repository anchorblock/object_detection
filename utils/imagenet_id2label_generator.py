### MAKE JSONS

from utils.imagenet_1k_classes  import IMAGENET2012_CLASSES
import json

id2label = {str(i): f"{key} - {value}" for i, (key, value) in enumerate(IMAGENET2012_CLASSES.items())}
label2id = {f"{key} - {value}" : str(i) for i, (key, value) in enumerate(IMAGENET2012_CLASSES.items())}


with open("configs/datasets/imagenet-1k-id2label.json", "w") as json_file:
    json.dump(id2label, json_file)

with open("configs/datasets/imagenet-1k-label2id.json", "w") as json_file:
    json.dump(label2id, json_file)
