# To-Do

## Future Releases

### New README.md Organization

- [x] writing and maintaining [Vision_Data_Guide.md](./Vision_Data_Guide.md) for step-by-step downloading and preprocessing popular vision datasets:
    - [x] Imagenet-1k classification data
    - [x] Object Detection (COCO) data
    - [x] panoptic Segmentation (COCO) data

- [x] Updating and correction on [README.md](./README.md)



### Train Models with COCO_panoptic using imagenet-pretrained backbones (remaining)

- [ ] writing [augmentation with preprocessing script](./utils/augmentations.py) for COCO panoptic task
    COCO panoptic Augmentation:

    while training:

    | Parameter                     | Value     |
    |-------------------------------|-----------|
    | large-scale jittering (LSJ)   | 0.1-2.0   |
    | fixed size crop               | 1024×1024 |

    while inference:

    | Parameter                     | Value     |
    |-------------------------------|-----------|
    | Resize: shorter side          | upto 800  |
    | Resize: longer side           | upto 1333 |



- [ ] writing [evaluation script](./utils/evaluation.py) function: compute_metrics_coco_panoptic() for COCO panoptic task

    - AP: Average Precision
    - PQ: Panoptic Quality
    - mIoU: Mean Intersection over Union

- [ ] writing [evaluation script](./utils/evaluation.py) function: compute_metrics_coco_instance() for COCO instance task

    - AP-m: Average Precision for instance segmentation masks
    - AP-m-50: Average Precision at IoU threshold of 0.50 for instance segmentation masks
    - AP-m-75: Average Precision at IoU threshold of 0.75 for instance segmentation masks

- [ ] writing [evaluation script](./utils/evaluation.py) function: compute_metrics_coco_bbox() for COCO bbox detection task

    - AP-b: Average Precision for bounding boxes
    - AP-b-50: Average Precision at IoU threshold of 0.50 for bounding boxes
    - AP-b-75: Average Precision at IoU threshold of 0.75 for bounding boxes


- [ ] training and finetuning script: writing train_panoptic_seg.py for training with suitable hyperparameters
    
    training hyperparamters with COCO panoptic:

    | Parameter                      | Value     |
    |-------------------------------|----------|
    | Batch Size                    | 16     |
    | Base Learning Rate            | 0.0001     |
    | Learning Rate Scheduler       | Step   |
    | Learning Rate Multiplier to the backbone       | 0.1   |
    | Learning Rate decay at      | 90% and 95% of total training steps   |
    | Learning Rate decay factor each time      | 10   |
    | Training Epochs (depending on small/large backbone gradually)              | 50-100      |
    | Optimizer                     | AdamW    |
    | Gradient Clip                 | 5.0      |
    | Weight Decay                  | 0.05     |


- [ ] writing evaluate_panoptic_seg.py for evaluation with validation data

- [ ] training_coco_panoptic bash command --> README.md
- [ ] evaluating_coco_panoptic bash command --> README.md
- [x] inference_coco_panoptic bash command --> README.md



### Some unresolved debugging issues and miscellaneous

All of the scripts have undergone thorough checks, with the exception of a few that were either due to time constraints or because the pipeline was developed on a non-GPU computer. The following tasks need to be debugged:

- [ ] COCO labeling id and bbox id are consistent. But there are complete inconsistency in between label ids and panoptic mask label ids. Have to debug the panoptic dataset issue.
    - [ ] COCO builder script
    - [ ] COCO augmentation script
- [ ] test imagenet training script with the presence of GPU, debug and fix error
- [ ] test COCO panoptic training script with the presence of GPU, debug and fix error
- [ ] create all processed parquet data of imagenet and coco in zip format, upload in s3


### Making Pretrained Backbone and Panoptic Segmentation models more dynamic

- [ ] modify backbone training pipeline for working with -
    - [ ] any huggingface model hub or local model path
    - [ ] any custom pytorch model
- [ ] modify panoptic training pipeline for working with -
    - [ ] any huggingface model hub or local model path
    - [ ] any newly defined / customized huggingface model child class
    - [ ] any custom pytorch model


### Extensions of this repository for different datasets benchmarking

- [ ] Pretraining Data (Imagenet22k)
- [ ] Object Detection (COCO)
- [ ] instance Segmentation (COCO)
- [ ] Semantic Segmentation (ADE20K)


### Extensions of this repository for more custom Object Detection Models

- [ ] Mask-RCNN-1x
- [ ] Mask-RCNN-3x


### Extensions of this repository for more Semantic Segmentation Models for benchmarking

- [ ] UperNet
- [ ] SegFormer


### Estimation of total training time, #Params(M) and GPU memory usage (5-10 iterations)

Results will be written in a table format: [Estimation_params_time_GPU_usage.md](./Estimation_params_time_GPU_usage.md)


- [ ] classification pretraining models
    - [ ] swin-tiny
    - [ ] swin-small
    - [ ] swin-base
    - [ ] focalnet-tiny
    - [ ] focalnet-small
    - [ ] focalnet-base

- [ ] classification models
    - [ ] swin-tiny
    - [ ] swin-small
    - [ ] swin-base
    - [ ] focalnet-tiny
    - [ ] focalnet-small
    - [ ] focalnet-base

- [ ] object detection models
    - [ ] Mask-RCNN-1x with backbones of - 
        - [ ] swin-tiny
        - [ ] swin-small
        - [ ] swin-base
        - [ ] focalnet-tiny
        - [ ] focalnet-small
        - [ ] focalnet-base
    - [ ] Mask-RCNN-3x with backbones of - 
        - [ ] swin-tiny
        - [ ] swin-small
        - [ ] swin-base
        - [ ] focalnet-tiny
        - [ ] focalnet-small
        - [ ] focalnet-base


- [ ] semantic segmentation models
    - [ ] UperNet with backbones of - 
        - [ ] swin-tiny
        - [ ] swin-small
        - [ ] swin-base
        - [ ] focalnet-tiny
        - [ ] focalnet-small
        - [ ] focalnet-base
    - [ ] SegFormer with backbones of - 
        - [ ] swin-tiny
        - [ ] swin-small
        - [ ] swin-base
        - [ ] focalnet-tiny
        - [ ] focalnet-small
        - [ ] focalnet-base
    - [ ] mask2former with backbones of - 
        - [ ] swin-tiny
        - [ ] swin-small
        - [ ] swin-base
        - [ ] focalnet-tiny
        - [ ] focalnet-small
        - [ ] focalnet-base

- [ ] instance segmentation models
    - [ ] DeTR with backbones of - 
        - [ ] swin-tiny
        - [ ] swin-small
        - [ ] swin-base
        - [ ] focalnet-tiny
        - [ ] focalnet-small
        - [ ] focalnet-base
    - [ ] mask2former with backbones of - 
        - [ ] swin-tiny
        - [ ] swin-small
        - [ ] swin-base
        - [ ] focalnet-tiny
        - [ ] focalnet-small
        - [ ] focalnet-base

- [ ] panoptic segmentation models
    - [ ] DeTR with backbones of - 
        - [ ] swin-tiny
        - [ ] swin-small
        - [ ] swin-base
        - [ ] focalnet-tiny
        - [ ] focalnet-small
        - [ ] focalnet-base
    - [ ] mask2former with backbones of - 
        - [ ] swin-tiny
        - [ ] swin-small
        - [ ] swin-base
        - [ ] focalnet-tiny
        - [ ] focalnet-small
        - [ ] focalnet-base





<br>




# Finished

## Pre-Alpha Release 0.1.0

### Train Backbones with ImageNet-1k (Partial)

- [x] **Repo organization:** Repository organization for training both with imagenet and coco for multiple backbone classifiers and panoptic segmentation models

    ```
        .
        ├── configs/
        ├── data/
        ├── dev/
        ├── formatted_data/
        ├── models/
        ├── outputs/
        ├── scripts/
        └── utils/
    ```
- [x] **Imagenet scripts (download & extract):** writing script: [imagenet_1k_download.sh](./data/imagenet_1k_download.sh), [imagenet_1k_extract.sh](./data/imagenet_1k_extract.sh)
- [x] **Raw data to Huggingface format:** writing [utils/imagenet_1k_dataset_script.py](./utils/imagenet_1k_dataset_script.py) similar to coco_script availale online for working with local imagenet files for getting image as huggingface datasets format. Sample:  ```{"image": <PIL Image>, "label": 6}```
- [x] **Imagenet label files preparation:** prepared imagenet label files and fixed label-related bugs: [configs/datasets/imagenet-1k-id2label.json](./configs/datasets/imagenet-1k-id2label.json), [configs/datasets/imagenet-1k-label2id.json](./configs/datasets/imagenet-1k-label2id.json)
- [x] **backbone models' configs:** general [config](./configs/backbones) and [preprocess_config](./configs/backbones) backbones: bit, convnext, convnextv2, dinat, focalnet, nat, resnet, swin
- [x] **Imagenet compression:** [scripts/raw_to_parquet_imagenet.py](./scripts/raw_to_parquet_imagenet.py) for faster saving and loading in parquet format (compressed ~147 GB imagenet data to ~11 MB by saving image in Pillow format inside parquet chart).
- [x] **Augmentation script** Augmentation script [utils/augmentations.py](./utils/augmentations.py) according to [FocalNet paper](https://arxiv.org/abs/2203.11926) imagenet augmentations

    | Parameter                      | Value     |
    |-------------------------------|----------|
    | Color Jitter Factor           | 0.4      |
    | Auto-augmentation             | rand-m9-mstd0.5-inc1 |
    | Random Erasing Probability    | 0.25     |
    | Random Erasing Mode           | Pixel    |
    | Mixup α                       | 0.8      |
    | Cutmix α                      | 0.8      |
    | Mixup Probability             | 1.0      |
    | Mixup Switch Probability      | 0.5      |
    | Label Smoothing               | 0.1      |


<br>


## Pre-Alpha Release 0.2.0

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



## Pre-Alpha Release 0.3.0

### (write something)


