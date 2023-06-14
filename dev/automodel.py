import sys
sys.path.append('./')

from transformers import MaskFormerImageProcessor #, AutoModelForUniversalSegmentation
from PIL import Image
import requests
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

print(predicted_panoptic_map)

import matplotlib.pyplot as plt
import torch

# Convert the tensor to a numpy array
image_array = predicted_panoptic_map.numpy()

# Display the image using Matplotlib
plt.imshow(image_array, cmap='gray')
plt.axis('off')
plt.show()