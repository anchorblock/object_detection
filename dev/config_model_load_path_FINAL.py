from transformers import DetrConfig, DetrForObjectDetection, FocalNetConfig, FocalNetModel, FocalNetForImageClassification


config = FocalNetConfig.from_json_file('./weights_focalnet_img_classify/config.json')

# config.id2label = {"0": "a", "1": "b"}

model = FocalNetForImageClassification.from_pretrained('./weights_focalnet_img_classify/pytorch_model.bin', config=config)

print(model)


######## dynamic load

backbone_name = "dinat"
model = getattr(config.backbones, backbone_name).model
