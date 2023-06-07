# object-detection v0.1.0-pre.alpha Release Notes

## Contributors:
- Tanzila BInti Alam

## Features - 

- **Repo organization:** Repository organization for training both with imagenet and coco for multiple backbone classifiers and panoptic segmentation models

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
- **Imagenet scripts (download & extract):** writing script: [imagenet_1k_download.sh](./data/imagenet_1k_download.sh), [imagenet_1k_extract.sh](./data/imagenet_1k_extract.sh)
- **Raw data to Huggingface format:** writing [utils/imagenet_1k_dataset_script.py](./utils/imagenet_1k_dataset_script.py) similar to coco_script availale online for working with local imagenet files for getting image as huggingface datasets format. Sample:  ```{"image": <PIL Image>, "label": 6}```
- **Imagenet label files preparation:** prepared imagenet label files and fixed label-related bugs: [configs/datasets/imagenet-1k-id2label.json](./configs/datasets/imagenet-1k-id2label.json), [configs/datasets/imagenet-1k-label2id.json](./configs/datasets/imagenet-1k-label2id.json)
- **backbone models' configs:** general [config](./configs/backbones) and [preprocess_config](./configs/backbones) backbones: bit, convnext, convnextv2, dinat, focalnet, nat, resnet, swin
- **Imagenet compression:** [scripts/raw_to_parquet_imagenet.py](./scripts/raw_to_parquet_imagenet.py) for faster saving and loading in parquet format (compressed ~147 GB imagenet data to ~11 MB by saving image in Pillow format inside parquet chart).
- **Augmentation script** Augmentation script [utils/augmentations.py](./utils/augmentations.py) according to [FocalNet paper](https://arxiv.org/abs/2203.11926) imagenet augmentations

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

