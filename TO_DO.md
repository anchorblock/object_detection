# To-Do

## Future Releases

### Train Backbones with ImageNet-1k (remaining)

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

    During fine-tuning, cutmix and mixup have to be disabled.


- [x] [scripts/evaluate_backbone_classifier.py](./scripts/evaluate_backbone_classifier.py) evaluation for imagenet validation data

- [x] training_imagenet_1k bash command --> [README.md](./README.md#🚀-training-backbones-with-imagenet-1k-and-config-files)
- [x] evaluation_imagenet_1k bash command --> [README.md](./README.md#📊-evaluate-backbones-with-imagenet-1k-validation-data)
- [x] inference python command --> [README.md](./README.md#💡-inference-with-backbones)


### Train Models with COCO_panoptic using imagenet-pretrained backbones

- [x] writing download script:  [coco_datasets_download.sh](./data/coco_datasets_download.sh), [coco_datasets_extract.sh](./data/coco_datasets_extract.sh)

- [ ] COCO label files preparation: [configs/datasets/coco-panoptic-id2label.json](./configs/datasets/coco-panoptic-id2label.json), [configs/datasets/coco-panoptic-label2id.json](./configs/datasets/coco-panoptic-label2id.json)


- [ ] Raw data to Huggingface format script: writing [utils/coco_dataset_script.py](./utils/coco_dataset_script.py) for working with local COCO datasets files for getting data as huggingface datasets format. 

    Sample:  {'image_id': 491000, 'caption_id': 3753, 'caption': 'Pedestrians walking down a sidewalk next to a small street.', 'height': 429, 'width': 640, 'file_name': '000000491000.jpg', 'coco_url': 'http://images.cocodataset.org/train2017/000000491000.jpg', 'image_path': 'data/coco_datasets/train2017/000000491000.jpg'}



- [ ] COCO data compression script

<!-- [scripts/raw_to_parquet_imagenet.py](./scripts/raw_to_parquet_imagenet.py) for faster saving and loading in parquet format (compressed ~147 GB imagenet data to ~11 MB by saving image in Pillow format inside parquet chart). -->


- [ ] general config and preprocess_config architectures: DeTR, maskformer, mask2former, oneformer
- [ ] raw model architecture script: DeTR.py
- [ ] raw model architecture script: maskformer.py
- [ ] raw model architecture script: mask2former.py
- [ ] raw model architecture script: oneformer.py

- [ ] training_script_coco_panoptic.py (load config from dict path, pretrained_weight_backbone_load_path, save_path (temporary))
- [ ] inference_script_coco_panoptic.ipynb
- [ ] training_coco_panoptic bash command (readme, trial)
- [ ] inference_coco_panoptic (readme, trial)



<br>

## Pre-Alpha Release 0.1.0

### Train Backbones with ImageNet-1k (Partial)

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


<br>

## Pre-Alpha Release 0.2.0

### Train Backbones with ImageNet-1k (Partial)

