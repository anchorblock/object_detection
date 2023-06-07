# To-Do

## Future Releases

### Train Backbones with ImageNet-1k (remaining)

- [ ] Evaluation Metrics script [utils/evaluation_metrics.py](./utils/evaluation_metrics.py) for popular evaluation metrics used for ImageNet-1K:

    - Top-1 Accuracy
    - Top-5 Accuracy
    - Precision
    - Recall
    - F1-Score
    - Confusion Matrix
    - Mean Average Precision (mAP)

- [ ] scripts/train_backbone_classifier.py (training script of backbone with imagenet data) with hparams according to [FocalNet paper](https://arxiv.org/abs/2203.11926) training with imagenet

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


- [ ] inference_script_backbone_classifier.ipynb
- [ ] training_imagenet_1k bash command --> README.md
- [ ] inference_imagenet_1k bash command --> README.md


### Train Models with COCO_panoptic using imagenet-pretrained backbones

- [x] writing download script: coco_datasets_download_and_extract.sh
- [ ] general config and preprocess_config architectures: DeTR
- [ ] general config and preprocess_config architectures: maskformer
- [ ] general config and preprocess_config architectures: mask2former
- [ ] general config and preprocess_config architectures: oneformer
- [ ] architecture model and preprocessor loading scripts: DeTR.py
- [ ] architecture model and preprocessor loading scripts: maskformer.py
- [ ] architecture model and preprocessor loading scripts: mask2former.py
- [ ] architecture model and preprocessor loading scripts: oneformer.py
- [ ] download_script: coco_download_and_extract.sh
- [ ] training_script_coco_panoptic.py (load config from dict path, pretrained_weight_backbone_load_path, save_path (temporary))
- [ ] inference_script_coco_panoptic.ipynb
- [ ] training_coco_panoptic bash command (readme, trial)
- [ ] inference_coco_panoptic (readme, trial)



<br>

## Pre-Alpha Release 0.1.0

### Train Backbones with ImageNet-1k (Partial)

- [x] Repository organization for training both with imagenet and coco for multiple backbone classifiers and panoptic segmentation models

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

- [x] writing download script: [imagenet_1k_download.sh](./data/imagenet_1k_download.sh), [imagenet_1k_extract.sh](./data/imagenet_1k_extract.sh)
- [x] writing [utils/imagenet_1k_dataset_script.py](./utils/imagenet_1k_dataset_script.py) similar to coco_script availale online for working with local imagenet files for getting image as huggingface datasets format. Sample:  ```{"image": <PIL Image>, "label": 6}```
- [x] prepared imagenet label files and fixed label-related bugs: [configs/datasets/imagenet-1k-id2label.json](./configs/datasets/imagenet-1k-id2label.json), [configs/datasets/imagenet-1k-label2id.json](./configs/datasets/imagenet-1k-label2id.json)
- [x] general [config](./configs/backbones) and [preprocess_config](./configs/backbones) backbones: bit, convnext, convnextv2, dinat, focalnet, nat, resnet, swin
- [x] [scripts/raw_to_parquet_imagenet.py](./scripts/raw_to_parquet_imagenet.py) for faster saving and loading in parquet format (compressed ~147 GB imagenet data to ~11 MB by saving image in Pillow format inside parquet chart).
- [x] Augmentation script [utils/augmentations.py](./utils/augmentations.py) according to [FocalNet paper](https://arxiv.org/abs/2203.11926) imagenet augmentations

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


