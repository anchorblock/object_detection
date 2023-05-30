# object_detection

Object detection benchmarking with coco dataset using different transformers backbones and different architectures for panoptic segmentation task. We are using tiny version of every models and every backbones for training and inference convenience.

## To-Do

Releases and To-Do archive lists, referring to [TO_DO.md](./TO_DO.md)


## Install Requirements

To setup and installation, referring to [INSTALL.md](./INSTALL.md)

## Backbones Training Pipelines

The following backbones can be trained with imagenet-1k dataset using this current repository:

backbones : 

- bit
- convnext
- convnextv2
- dinat
- focalnet
- nat
- resnet
- swin


### Download ImageNet dataset and format for training

For using the ImageNet-1k dataset (2012), you need to download, extract and organize it manually first.

Download:

```bash
bash imagenet_1k_download.sh
```

Extract:

```bash
bash imagenet_1k_extract.sh
```


For testing purpose, you can use the hosted dataset as follows:

```python
import datasets

IMAGENET_DIR = "imagenet_1k"
ds = datasets.load_dataset("utils/imagenet_1k_dataset_script.py", data_dir=IMAGENET_DIR)
ds["train"][0]
```













### Train backbones using ImageNet and config files

To train classifier model, run:

<!-- 


```bash
export OMP_NUM_THREADS=4
export n_gpu=1

torchrun --standalone --nproc_per_node=$n_gpu z05_training_fp16_DDP_A100.py \
    --train_parquet_data_file="./mlm_processed_bn_data/train_data.parquet" \
    --test_parquet_data_file="./mlm_processed_bn_data/validation_data.parquet" \
    --per_device_train_batch_size=32 \
    --gradient_accumulation_steps=64 \
    --learning_rate=5e-4 \
    --warmup_steps=10000 \
    --max_steps=1250000 \
    --logging_steps=500 \
    --eval_steps=100000 \
    --save_steps=10000 \
    --init_model_directory="./DeBERTaV3" \
    --save_directory="./DeBERTaV3_trained_bn" \
    --resume_from_checkpoint="./DeBERTaV3" \
    --gradient_checkpointing=true

```

SNIPPET !

export OMP_NUM_THREADS=4
export n_gpu=1

# Create a config file
cat << EOF > config.yaml
train_parquet_data_file: "./mlm_processed_bn_data/train_data.parquet"
test_parquet_data_file: "./mlm_processed_bn_data/validation_data.parquet"
per_device_train_batch_size: 32
gradient_accumulation_steps: 64
learning_rate: 5e-4
warmup_steps: 10000
max_steps: 1250000
logging_steps: 500
eval_steps: 100000
save_steps: 10000
init_model_directory: "./DeBERTaV3"
save_directory: "./DeBERTaV3_trained_bn"
resume_from_checkpoint: "./DeBERTaV3"
gradient_checkpointing: true
EOF

# Pass the arguments mentioned in the config file
torchrun --standalone --nproc_per_node=$n_gpu z05_training_fp16_DDP_A100.py \
    --config config.yaml



 -->













## Download COCO dataset and format for training

For using the COCO dataset (2017), you need to download and extract it manually first:

Download:

```bash
bash coco_datasets_download.sh
```

Extract:

```bash
bash coco_datasets_extract.sh
```


Expected dataset structure for COCO:

```
coco_datasets/
  annotations/
    instances_{train,val}2017.json
    panoptic_{train,val}2017.json
    caption_{train,val}2017.json
    # evaluate on instance labels derived from panoptic annotations
    panoptic2instances_val2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
  panoptic_{train,val}2017/  # png annotations
  panoptic_semseg_{train,val}2017/  # generated by the script mentioned below
```

For testing purpose, you can use the hosted dataset as follows:

```python
import datasets

COCO_DIR = "coco_datasets"
ds = datasets.load_dataset("utils/coco_dataset_script.py", "2017", data_dir=COCO_DIR)
ds["train"][0]
```

## Train different models for different backbones

To train model, firstly different image classification models will be pretrained with imagenet. Then, panoptic segmentation model will be built based on pre-trained backbone. 

(later ...)


## Supported Backbones for Architectures

List of supported backbones - bit, convnext, convnextv2, dinat, focalnet, maskformer-swin, nat, resnet, swin.


- **maskformer, mask2former** use detr object detection architecture as decoder, and are not dynamic for changing decoder architecture (facebook)
- **maskformer, mask2former** are currently only supporting maskformer-swin-transformer (not vanilla swin-transformer) as backbone (facebook). Any change in maskformer/mask2former backbone requires new architecture design.
- **oneformer** supports only above mentioned backbones/ classifiers.
- **DeTR** supports only above mentioned backbones/ classifiers.



## References & Citations

ImageNet Datasets:
```
@inproceedings{deng2009imagenet,
  title={Imagenet: A large-scale hierarchical image database},
  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  booktitle={2009 IEEE conference on computer vision and pattern recognition},
  pages={248--255},
  year={2009},
  organization={Ieee}
}
```

COCO Datatsets:

```
@article{DBLP:journals/corr/LinMBHPRDZ14,
  author    = {Tsung{-}Yi Lin and
               Michael Maire and
               Serge J. Belongie and
               Lubomir D. Bourdev and
               Ross B. Girshick and
               James Hays and
               Pietro Perona and
               Deva Ramanan and
               Piotr Doll{'{a} }r and
               C. Lawrence Zitnick},
  title     = {Microsoft {COCO:} Common Objects in Context},
  journal   = {CoRR},
  volume    = {abs/1405.0312},
  year      = {2014},
  url       = {http://arxiv.org/abs/1405.0312},
  archivePrefix = {arXiv},
  eprint    = {1405.0312},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/LinMBHPRDZ14},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
