# object-detection v0.2.0-pre.alpha

Tanzila Binti Alam -
- Backbone Classification Task:
    - Evaluation Metrics Script, Backbone Training Script, Backbone Evaluation Script, backbone training bash command with hyperparameters, backbone evaluation bash command, Sample inference code
- Panoptic Segmentation Task
    - COCO bash scripts (download & extract), Raw data to Huggingface format builder script, COCO compression, panoptic models' configs, ```./models/__init__.py``` as package, custom AutoPanopticConfig and AutoModelForPanopticSegmentation class, panoptic models' configs



Contribution:

### Train Backbones with ImageNet-1k (Remaining)

- [x] **Evaluation Metrics Script:** Evaluation Metrics script [utils/evaluation.py](./utils/evaluation.py) for popular evaluation metrics used for ImageNet-1K:

    - Top-1 Accuracy
    - Top-5 Accuracy
    - Precision
    - Recall
    - F1-Score
    - Mean Average Precision (mAP)

- [x] **Backbone Training Script:** [scripts/train_backbone_classifier.py](./scripts/train_backbone_classifier.py) (training script of backbone with imagenet data) with hparams according to [FocalNet paper](https://arxiv.org/abs/2203.11926) training with imagenet

    | Parameter                      | Value     |
    |-------------------------------|----------|
    | Batch Size                    | 1024     |
    | Base Learning Rate            | 1e-3     |
    | Learning Rate Scheduler       | Cosine   |
    | Minimum Learning Rate         | 1e-5     |
    | Warm-up Epochs                | 20       |
    | Training Epochs               | 300      |
    | Finetuning Epochs             | 30       |
    | Warm-up Schedule              | Linear   |
    | Warm-up Learning Rate         | 1e-6     |
    | Optimizer                     | AdamW    |
    | Stochastic Drop Path Rate     | 0.2/0.3/0.5 |
    | Gradient Clip                 | 5.0      |
    | Weight Decay                  | 0.05     |

    During fine-tuning epochs, cutmix and mixup will be disabled automatically, and it will be handleed with custom callback class.


- [x] **Backbone Evaluation Script:** [scripts/evaluate_backbone_classifier.py](./scripts/evaluate_backbone_classifier.py) evaluation for imagenet validation data

- [x] **backbone training bash command with hyperparameters:** training_imagenet_1k bash command --> [README.md](./README.md#🚀-training-backbones-with-imagenet-1k-and-config-files)
- [x] **backbone evaluation bash command:** evaluation_imagenet_1k bash command --> [README.md](./README.md#📊-evaluate-backbones-with-imagenet-1k-validation-data)
- [x] **Sample inference code:** inference python command --> [README.md](./README.md#💡-inference-with-backbones)


### Train Models with COCO_panoptic using imagenet-pretrained backbones (Partial)

- [x]  **COCO bash scripts (download & extract):** writing download script:  [coco_datasets_download.sh](./data/coco_datasets_download.sh), [coco_datasets_extract.sh](./data/coco_datasets_extract.sh)

- [x] **COCO label files preparation:**
    - [x] [configs/datasets/coco-detection-id2label.json](./configs/datasets/coco-detection-id2label.json)
    - [x] [configs/datasets/coco-detection-label2id.json](./configs/datasets/coco-detection-label2id.json)
    - [x] [configs/datasets/coco-panoptic-id2label.json](./configs/datasets/coco-panoptic-id2label.json)
    - [x] [configs/datasets/coco-panoptic-label2id.json](./configs/datasets/coco-panoptic-label2id.json)


- [x] **Raw data to Huggingface format:** Huggingface builder script: writing [utils/coco_dataset_script.py](./utils/coco_dataset_script.py) dynamic Builder Classes for working with local COCO datasets files for getting data as huggingface datasets format. This script is dynamic; following types of COCO data can be built with this script: "2017_detection", "2017_panoptic", "2017_detection_skip", "2017_panoptic_skip". 

    Sample:  {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x425 at 0x7F35BE2EA5B0>, 'image/filename': '000000000034.jpg', 'image/id': 34, 'panoptic_objects': [{'id': 5069153, 'area': 92893, 'bbox': [1, 20, 442, 399], 'is_crowd': False, 'category_id': 24, 'category_name': 'zebra', 'supercategory_id': 3, 'supercategory_name': 'animal', 'is_thing': True}, {'id': 2589299, 'area': 177587, 'bbox': [0, 0, 640, 425], 'is_crowd': False, 'category_id': 193, 'category_name': 'grass-merged', 'supercategory_id': 17, 'supercategory_name': 'plant', 'is_thing': False}], 'panoptic_image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=640x425 at 0x7F35BE2EA730>, 'panoptic_image/filename': '000000000034.png'}


- [x] **COCO compression:** COCO data compression script: [scripts/raw_to_parquet_coco.py](./scripts/raw_to_parquet_coco.py) for faster saving and loading in parquet format (compressed full ~28.2 GB COCO data to total ~133 MB by saving images in Pillow format inside parquet chart).


- [x] **panoptic models' configs:** panoptic model architectures: pretrained backbone weight can be loaded, and supported for these backbones: bit, convnext, convnextv2, dinat, focalnet, nat, resnet, swin
    - [x] [detr.py](./models_panoptic/detr.py) custom model class script
    - [x] [maskformer.py](./models_panoptic/maskformer.py) custom model class script
    - [x] [mask2former.py](./models_panoptic/mask2former.py) custom model class script
    - [x] [custom_maskformer.py](./models_panoptic/custom_maskformer.py) custom model class script
    - [x] [custom_mask2former.py](./models_panoptic/custom_mask2former.py) custom model class script
    - [x] [oneformer.py](./models_panoptic/oneformer.py) custom model class script


- [x] **./models/__init__.py as package:** writing [models_panoptic/__init__.py](./models_panoptic/__init__.py) for importing modules

- [x] **AutoModelForPanopticSegmentation class:** writing [models/automodel_panoptic.py](./models/automodel_panoptic.py) for customly defined modules for working with automodel class "AutoModelForPanopticSegmentation"

- [x] **AutoPanopticConfig class:** writing [models/autoconfig_panoptic.py](./models/autoconfig_panoptic.py) for customly defined modules for working with all customized panoptic config classs: "AutoPanopticConfig"


- [x] **panoptic models' configs:** general config and preprocess_config architectures: 
    - [x] [DeTR](./configs/architectures/detr), 
    - [x] [maskformer](./configs/architectures/maskformer), 
    - [x] [mask2former](./configs/architectures/mask2former),
    - [x] [custom_maskformer](./configs/architectures/custom_maskformer),
    - [x] [custom_mask2former](./configs/architectures/custom_mask2former),
    - [x] [oneformer](./configs/architectures/oneformer); and change "architectures" parameter with custom class name

