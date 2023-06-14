# To-Do

## Future Releases

### Train Models with COCO_panoptic using imagenet-pretrained backbones (remaining)


- [ ] general config and preprocess_config architectures: [DeTR](./configs/architectures/DeTR), [maskformer](./configs/architectures/maskformer), [mask2former](./configs/architectures/mask2former), [oneformer](./configs/architectures/oneformer); and change "architectures" parameter with custom class name

- [ ] writing [augmentation script](./utils/augmentations.py) for COCO panoptic task
- [ ] writing [evaluation script](./utils/evaluation.py) for COCO panoptic task

- [ ] training and finetuning script: writing train_panoptic_seg.py for training with suitable hyperparameters
- [ ] writing evaluate_panoptic_seg.py for evaluation

- [ ] training_coco_panoptic bash command --> README.md
- [ ] evaluating_coco_panoptic bash command --> README.md
- [x] inference_coco_panoptic bash command --> README.md



### Debugging training scripts with GPU

- [x] create all processed parquet data of imagenet and coco in zip format, upload in s3 and add download refernce in README.md
- [ ] test imagenet training script with the presence of GPU, debug and fix error
- [ ] test COCO panoptic training script with the presence of GPU, debug and fix error



### Making Pretrained Backbone and Panoptic Segmentation models more dynamic

- [ ] modify backbone training pipeline for working with -
    - [ ] any huggingface model hub or local model path
    - [ ] any custom pytorch model
- [ ] modify panoptic training pipeline for working with -
    - [ ] any huggingface model hub or local model path
    - [ ] any newly defined / customized huggingface model child class
    - [ ] any custom pytorch model

<br>


 ----------------------------




<br>










## Pre-Alpha Release 0.1.0

### Train Backbones with ImageNet-1k (Partial)

- [x] **Repo organization:** Repository organization for training both with imagenet and coco for multiple backbone classifiers and panoptic segmentation models

    ```
        .
        ├── configs/
        ├── data/
        ├── dev/
        ├── formatted_data/
        ├── models_panoptic/
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


- [x] Evaluation Metrics script [utils/evaluation.py](./utils/evaluation.py) for popular evaluation metrics used for ImageNet-1K:

    - Top-1 Accuracy
    - Top-5 Accuracy
    - Precision
    - Recall
    - F1-Score
    - Mean Average Precision (mAP)

- [x] [scripts/train_backbone_classifier.py](./scripts/train_backbone_classifier.py) (training script of backbone with imagenet data) with hparams according to [FocalNet paper](https://arxiv.org/abs/2203.11926) training with imagenet

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


- [x] [scripts/evaluate_backbone_classifier.py](./scripts/evaluate_backbone_classifier.py) evaluation for imagenet validation data

- [x] training_imagenet_1k bash command --> [README.md](./README.md#🚀-training-backbones-with-imagenet-1k-and-config-files)
- [x] evaluation_imagenet_1k bash command --> [README.md](./README.md#📊-evaluate-backbones-with-imagenet-1k-validation-data)
- [x] inference python command --> [README.md](./README.md#💡-inference-with-backbones)


### Train Models with COCO_panoptic using imagenet-pretrained backbones (Partial)

- [x] writing download script:  [coco_datasets_download.sh](./data/coco_datasets_download.sh), [coco_datasets_extract.sh](./data/coco_datasets_extract.sh)

- [x] COCO label files preparation: 
    - [x] [configs/datasets/coco-detection-id2label.json](./configs/datasets/coco-detection-id2label.json)
    - [x] [configs/datasets/coco-detection-label2id.json](./configs/datasets/coco-detection-label2id.json)
    - [x] [configs/datasets/coco-panoptic-id2label.json](./configs/datasets/coco-panoptic-id2label.json)
    - [x] [configs/datasets/coco-panoptic-label2id.json](./configs/datasets/coco-panoptic-label2id.json)


- [x] Huggingface builder script: writing [utils/coco_dataset_script.py](./utils/coco_dataset_script.py) dynamic Builder Classes for working with local COCO datasets files for getting data as huggingface datasets format. This script is dynamic; following types of COCO data can be built with this script: "2017_detection", "2017_panoptic", "2017_detection_skip", "2017_panoptic_skip". 

    Sample:  {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x425 at 0x7F35BE2EA5B0>, 'image/filename': '000000000034.jpg', 'image/id': 34, 'panoptic_objects': [{'id': 5069153, 'area': 92893, 'bbox': [1, 20, 442, 399], 'is_crowd': False, 'category_id': 24, 'category_name': 'zebra', 'supercategory_id': 3, 'supercategory_name': 'animal', 'is_thing': True}, {'id': 2589299, 'area': 177587, 'bbox': [0, 0, 640, 425], 'is_crowd': False, 'category_id': 193, 'category_name': 'grass-merged', 'supercategory_id': 17, 'supercategory_name': 'plant', 'is_thing': False}], 'panoptic_image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=640x425 at 0x7F35BE2EA730>, 'panoptic_image/filename': '000000000034.png'}


- [x] COCO data compression script: [scripts/raw_to_parquet_coco.py](./scripts/raw_to_parquet_coco.py) for faster saving and loading in parquet format (compressed full ~28.2 GB COCO data to total ~133 MB by saving images in Pillow format inside parquet chart).


- [x] architecture: [DeTR.py](./models_panoptic/DeTR.py) custom model class script
- [x] architecture: [maskformer.py](./models_panoptic/maskformer.py) custom model class script
- [x] architecture: [mask2former.py](./models_panoptic/mask2former.py) custom model class script
- [x] architecture: [oneformer.py](./models_panoptic/oneformer.py) custom model class script

- [x] writing [models_panoptic/__init__.py](./models_panoptic/__init__.py) for importing modules

- [x] writing [models/automodel_panoptic.py](./models/automodel_panoptic.py) for customly defined modules for working will automodel class "AutoModelForPanopticSegmentation"