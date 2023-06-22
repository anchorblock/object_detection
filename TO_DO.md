# To-Do

## Future Releases

### Newly organized repo

- [ ] `./notebooks` folder preparation and write all trial notebooks and inference scripts (both backbones and coco detection, semantic, instance, panoptic)
- [ ] `./docs` folder preparation and add all markdown files, complete all the markdown files' writings and correction of dependenble paths
- [ ] write new / updated benchmarking plan and add in `./docs/Estimation_params_time_GPU_usage.md`

- [ ] writw and maintain `./docs/Vision_Data_Guide.md` for step-by-step downloading and preprocessing popular vision datasets:
    - [ ] Imagenet-1k classification data
    - [ ] Object Detection (COCO) data
    - [ ] panoptic Segmentation (COCO) data

### Changing Backbone Models' organization and training pipeline

The object_detection repo became too big and messy; needs restructuring to address its size and complexity. Adding a new model currently requires modifying numerous scripts (around 11 scripts), which complicates the process. To simplify and improve scalability, the entire repository, model files, loading scheme, and dependencies need to be organized and edited accordingly.

- [ ] change `./configs/backbones` folder entirely by introducing a new organization approach with 
    - local loading script, 
    - local config files, 
    - checkpoint loading scheme, 
    - add autotype key and 
    - maintain separate .py script for every model and config class 
    
    (if any; especially for custom backbone models; these extra scripts are not required for huggingface hub's models). If this method maintains, we can train any local or huggingface hub's model with just one script.

- [ ] change `./scripts/train_backbone_classifier.py` backbone training script's argument parsers and simplify loading method


### COCO_panoptic datasets preprocessing (correction, major bug fix)

- [ ] corrections on coco panoptic mask values: following scripts will be edited:
    - [ ] [configs/datasets/coco*](./configs/datasets/)
    - [ ] [coco builder script](./utils/coco_dataset_script.py)
    - [ ] [augmentation script](./utils/augmentations.py)

- [ ] develop preprocessing pipeline: `./notebooks/preprocessing_forward_pass_panoptic.ipynb`
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


### Changing Panoptic Models' organization

- [ ] change `./configs/architectures` folder entirely by introducing a new organization approach with 
    - local loading script, 
    - local config files, 
    - checkpoint loading scheme, 
    - add autotype key and 
    - maintain separate .py script for every model and config class 
    
    (if any; especially for custom panoptic/detection models, say mask-RCNN-3x; these extra scripts are not required for huggingface hub's models). If this method maintains, we can train any local or huggingface hub's model with just one script.


### Rewrite README.md

- [ ] Have to rewrite `./README.md` for new repo organization and correct bash commands and dependency paths for all the changed scripts

### Train Models with COCO_panoptic using imagenet-pretrained backbones

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


### Some untested debugging scripts in the presence of GPU and miscellaneous

All of the scripts have undergone thorough checks, with the exception of a few, because the pipeline was developed on a non-GPU computer. The following tasks need to be debugged:

- [ ] test imagenet training script with the presence of GPU, debug and fix error
- [ ] test COCO panoptic training script with the presence of GPU, debug and fix error
- [ ] create all processed parquet data of imagenet and coco in zip format, upload in s3


### Adding Custom Object Detection Models, data preparation and training pipeline

- [ ] Mask-RCNN-1x
- [ ] Mask-RCNN-3x


### Extensions of this repository for Adding More Semantic Segmentation Models for benchmarking

- [ ] UperNet
- [ ] SegFormer

### Extensions of this repository for different datasets benchmarking

- [ ] Pretraining Data (Imagenet22k)
- [ ] Object Detection (COCO)
- [ ] instance Segmentation (COCO)
- [ ] Semantic Segmentation (ADE20K)

### Estimation of total training time, #Params(M) and GPU memory usage (5-10 iterations)

Results will be written in a table format: `./docs/Estimation_params_time_GPU_usage.md`


<br>


# Finished

## Pre-Alpha Release 0.3.0

(add something)

<!-- ### DOCS Organization -->

