import sys
sys.path.append('./')

from transformers import AutoImageProcessor
from PIL import Image, ImageDraw
import numpy as np
import requests
import torch
from models import AutoModelForPanopticSegmentation


# load MaskFormer fine-tuned on COCO panoptic segmentation
feature_extractor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-tiny-coco")
model = AutoModelForPanopticSegmentation.from_pretrained("facebook/maskformer-swin-tiny-coco")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)

# you can pass them to feature_extractor for postprocessing
result = feature_extractor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
# we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)


predicted_panoptic_map = result["segmentation"]

# Get segments_info
segments_info = result['segments_info']


# Convert the tensor to numpy
image_array = predicted_panoptic_map.numpy()

# Normalize the array to the range 0-255
normalized_array = (image_array - np.min(image_array)) * (255 / (np.max(image_array) - np.min(image_array)))

# Convert the array to uint8 data type
uint8_array = normalized_array.astype(np.uint8)

# Create a PIL image from the uint8 array
image_gen_mask = Image.fromarray(uint8_array)

# Load the labels dictionary from the model configuration (model.config.id2label)
id2label = model.config.id2label


# Save the image
image_gen_mask.save('predicted_inference_map.png')
