import sys
sys.path.append('./')

from transformers import MaskFormerImageProcessor #, AutoModelForUniversalSegmentation
from PIL import Image, ImageDraw
import numpy as np
import requests
import torch
from models import AutoModelForPanopticSegmentation


# load MaskFormer fine-tuned on COCO panoptic segmentation
feature_extractor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-tiny-coco")
model = AutoModelForPanopticSegmentation.from_pretrained("facebook/maskformer-swin-tiny-coco")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)



# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# you can pass them to feature_extractor for postprocessing
result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
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
image = Image.fromarray(uint8_array)

# Load the labels dictionary from the model configuration (model.config.id2label)
id2label = model.config.id2label


# Create a PIL draw object
draw = ImageDraw.Draw(image)

# Iterate over the segments_info dictionary
for segment in segments_info:
    segment_id = segment['id']
    label_id = segment['label_id']
    label = id2label[label_id]
    score = segment['score']
    
    # Get the bounding box coordinates for the segment
    bbox = np.argwhere(image_array == segment_id)
    ymin, xmin = np.min(bbox, axis=0)
    ymax, xmax = np.max(bbox, axis=0)
    
    # Draw the bounding box rectangle
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='white')
    
    # Add label text
    text = f"{label} ({score:.2f})"
    draw.text((xmin, ymin - 12), text, fill='white')


# Save the image
image.save('predicted_panoptic_map.png')