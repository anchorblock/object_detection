from transformers import DetrConfig, DetrForObjectDetection, FocalNetConfig, FocalNetModel, FocalNetForImageClassification

focal_config = FocalNetConfig()

focal_model_pre = FocalNetForImageClassification(focal_config)

focal_model_pre.save_pretrained("./weights_focalnet_img_classify")

############################################################### convert to detr

focal_load_model = FocalNetModel.from_pretrained("./weights_focalnet_img_classify")


detr_configuration = DetrConfig(use_timm_backbone = False,
                           backbone_config=FocalNetConfig())

model = DetrForObjectDetection(detr_configuration)

print(model)

model.model.backbone.conv_encoder.model.focalnet.load_state_dict(focal_load_model.state_dict())


############################################################### inference

from transformers import AutoImageProcessor, DetrForObjectDetection

import torch

from PIL import Image

import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"

image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = image_processor(images=image, return_tensors="pt")

outputs = model(**inputs)


print(outputs)

# convert outputs (bounding boxes and class logits) to COCO API

target_sizes = torch.tensor([image.size[::-1]])

results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[

    0

]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):

    box = [round(i, 2) for i in box.tolist()]

    print(

        f"Detected {model.config.id2label[label.item()]} with confidence "

        f"{round(score.item(), 3)} at location {box}"

    )