## To-Do

### Pre-Alpha Release 0.1.0

#### Train Backbones with ImageNet-1k (Partial)

- [x] Repository organization
- [x] writing download script: imagenet_1k_download.sh, imagenet_1k_extract.sh
- [x] writing utils/imagenet_1k_dataset_script.py similar to coco_script availale online for working with local imagenet files for getting image as huggingface datasets format
- [x] general config and preprocess_config backbones: bit, convnext, convnextv2, dinat, focalnet, nat, resnet, swin
- [x] scripts/raw_to_parquet_imagenet.py for faster saving and loading

<br>

### Future Releases

#### Train Backbones with ImageNet-1k (remaining)

- [ ] Augmentation script utils/augmentations.py according to [FocalNet paper](https://arxiv.org/abs/2203.11926) imagenet augmentations using timm library

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
    | Stochastic Drop Path Rate     | 0.2/0.3/0.5 |
    | Label Smoothing               | 0.1      |


- [ ] scripts/train_backbone_classifier.py (training script of backbone with imagenet data) with hparams according to [FocalNet paper](https://arxiv.org/abs/2203.11926) training with imagenet

    | Parameter                      | Value     |
    |-------------------------------|----------|
    | Batch Size                    | 1024     |
    | Base Learning Rate            | 1e-3     |
    | Learning Rate Scheduler       | Cosine   |
    | Minimum Learning Rate         | 1e-5     |
    | Training Epochs               | 300      |
    | Warm-up Epochs                | 20       |
    | Warm-up Schedule              | Linear   |
    | Warm-up Learning Rate         | 1e-6     |
    | Optimizer                     | AdamW    |
    | Gradient Clip                 | 5.0      |
    | Weight Decay                  | 0.05     |


- [ ] inference_script_backbone_classifier.ipynb
- [ ] training_imagenet_1k bash command --> README.md
- [ ] inference_imagenet_1k bash command --> README.md


#### Train Models with COCO_panoptic using imagenet-pretrained backbones

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

