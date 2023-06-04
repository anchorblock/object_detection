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
bash data/imagenet_1k_download.sh
```

Extract:

```bash
bash data/imagenet_1k_extract.sh
```


For testing purpose, you can use the hosted dataset as follows:

```python
import datasets

IMAGENET_DIR = "data/imagenet_1k"
ds = datasets.load_dataset("utils/imagenet_1k_dataset_script.py", data_dir=IMAGENET_DIR, splits = ["validation", "test"], cache_dir=".cache")
ds["validation"][0]
```

An example output:

```python
>>> ds["validation"][5678]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x375 at 0x7F15A02B0B50>, 'label': 118}
```


### Train backbones using ImageNet-1k and config files

convert all raw data to huggingface image classification data format and save to parquet for faster loading:

```bash
python scripts/raw_to_parquet_imagenet.py \
    --imagenet_dir="data/imagenet_1k" \
    --save_path="formatted_data/imagenet_1k"
```

train classifier:

default hyperparameters for training (from [FocalNet paper](https://arxiv.org/abs/2203.11926)):


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
| Gradient Clip                 | 5.0      |
| Weight Decay                  | 0.05     |



<!-- 
inference:





 -->





























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
bash data/coco_datasets_download.sh
```

Extract:

```bash
bash data/coco_datasets_extract.sh
```


Expected dataset structure for COCO:

```
data/coco_datasets/
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

COCO_DIR = "data/coco_datasets"
ds = datasets.load_dataset("utils/coco_dataset_script.py", "2017", data_dir=COCO_DIR)
ds["train"][0]
```

## Train different models for different backbones

To train model, firstly different image classification models (backbones) will be pretrained with imagenet. Then, panoptic segmentation model will be built based on pre-trained backbone. 

Will be added in future release.


## Supported Backbones for Architectures

List of supported backbones - bit, convnext, convnextv2, dinat, focalnet, maskformer-swin, nat, resnet, swin.


- **maskformer, mask2former** only supporting detr object detection architecture as decoder.
- **maskformer, mask2former** are currently only supporting swin-transformer as backbone (facebook). Any change in maskformer/mask2former backbone requires new architecture design.
- **oneformer** supports only above mentioned backbones/ classifiers.
- **DeTR** supports only above mentioned backbones/ classifiers.


## References & Citations

ImageNet Datasets:
```
@article{imagenet15russakovsky,
    Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = { {ImageNet Large Scale Visual Recognition Challenge} },
    Year = {2015},
    journal   = {International Journal of Computer Vision (IJCV)},
    doi = {10.1007/s11263-015-0816-y},
    volume={115},
    number={3},
    pages={211-252}
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
