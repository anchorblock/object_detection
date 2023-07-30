# Step-by-Step: Downloading and Preprocessing Vision Datasets

Welcome to **Step-by-Step: Downloading and Preprocessing Vision Datasets!** In this guide, we will take you through the essential steps and techniques involved in acquiring and preparing vision datasets for your projects. Let's get started!

We will download and preprocess the following datasets:

 - [ImageNet-22k classification pretraining data](#imagenet-22k-classification-pretraining-data)
 - [Imagenet-1k classification data](#imagenet-1k-classification-data)
 - [COCO object detection data](#coco-object-detection-data)
 - [COCO panoptic segmentation data](#coco-panoptic-segmentation-data)
 - [COCO instance segmentation data](#coco-instance-segmentation-data)
 - [ADE20K semantic segmentation data](#ade20k-semantic-segmentation-data)

Let's dive deeper!

<br>

## Table of Contents

Check this out!

- [Vision Data Guide: Step-by-Step](#step-by-step-downloading-and-preprocessing-vision-datasets)
  - [Table of Contents](#table-of-contents)
  - [ImageNet-22k classification pretraining data](#imagenet-22k-classification-pretraining-data)
  - [Imagenet-1k classification data](#imagenet-1k-classification-data)
  - [COCO object detection data](#coco-object-detection-data)
  - [COCO panoptic segmentation data](#coco-panoptic-segmentation-data)
  - [COCO instance segmentation data](#coco-instance-segmentation-data)
  - [ADE20K semantic segmentation data](#ade20k-semantic-segmentation-data)
  - [Download all processed data at once from s3 bucket](#download-all-processed-data-at-once-from-s3-bucket)



<br>

## ImageNet-22k classification pretraining data

ImageNet-22k classification pretraining data is a large dataset containing over 22,000 categories for training deep learning models in image classification tasks. It serves as a valuable resource for pretraining models on a wide range of visual concepts, allowing them to learn general representations before fine-tuning on specific datasets.


🚧 ImageNet-22k classification pretraining data downloading and preprocessing steps will be added in future release. Stay tuned!

<br>

## Imagenet-1k classification data

Imagenet-1k classification data is a subset of the ImageNet dataset, focusing on image classification tasks. It consists of over 1,000 categories and serves as a benchmark for evaluating the performance of deep learning models in image classification. Each image is labeled with a single category, making it suitable for training and evaluating models' ability to assign the correct class to a given image.

<br>

### Downloading and Formatting the ImageNet Dataset

Install the following dependencies:

```bash
sudo apt update
sudo apt install axel
```

To use the ImageNet-1k dataset (2012), you need to manually download, extract, and organize it. Follow these steps:

1. Download the dataset:

```bash
bash data/imagenet_1k_download.sh
```

2. Extract the dataset:

```bash
bash data/imagenet_1k_extract.sh
```

For testing purposes, you can use the hosted dataset by executing the following Python code:

```python
import datasets

IMAGENET_DIR = "data/imagenet_1k"
ds = datasets.load_dataset("utils/dataset_utils/imagenet_1k_dataset_script.py", data_dir=IMAGENET_DIR, splits = ["validation", "test"], cache_dir=".cache")
ds["validation"][0]
```

An example output:

```python
>>> ds["validation"][5678]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x375 at 0x7F15A02B0B50>, 'label': 118}
```

Now, convert all raw data to huggingface image classification data format and save to parquet for faster loading:

```bash
python scripts/data_processing_scripts/raw_to_parquet_imagenet.py \
    --imagenet_dir="data/imagenet_1k" \
    --save_path="formatted_data/imagenet_1k"
```

<br>

## COCO object detection data

COCO object detection data is a widely used dataset for object detection tasks. It contains a diverse collection of images with complex scenes and multiple objects, annotated with bounding boxes and object categories. This dataset enables the training and evaluation of models to accurately detect and classify objects in images, making it valuable for advancing computer vision research and applications.

### Downloading and Formatting the COCO dataset (2017)


For using the COCO dataset (2017), you need to download and extract it manually first:

1. Download the dataset:

```bash
bash data/coco_datasets_download.sh
```

2. Extract the dataset:

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
```

For testing purposes, you can use the hosted dataset by executing the following Python code:

```python
import datasets

COCO_DIR = "data/coco_datasets"

# bbox_mode = one of ["corners", "height_width"]
# data_variant = one of ["2017_detection", "2017_panoptic", "2017_detection_skip", "2017_panoptic_skip"]

bbox_mode = "corners"
data_variant = "2017_panoptic"

ds = datasets.load_dataset("utils/dataset_utils/coco_dataset_script.py", data_variant, bbox_mode = bbox_mode, data_dir=COCO_DIR)
ds["train"][0]
```

An example output:

```python
>>> ds["train"][3]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x425 at 0x7F35BE2EA5B0>, 'image/filename': '000000000034.jpg', 'image/id': 34, 'panoptic_objects': [{'id': 5069153, 'area': 92893, 'bbox': [1, 20, 442, 399], 'is_crowd': False, 'category_id': 24, 'category_name': 'zebra', 'supercategory_id': 3, 'supercategory_name': 'animal', 'is_thing': True}, {'id': 2589299, 'area': 177587, 'bbox': [0, 0, 640, 425], 'is_crowd': False, 'category_id': 193, 'category_name': 'grass-merged', 'supercategory_id': 17, 'supercategory_name': 'plant', 'is_thing': False}], 'panoptic_image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=640x425 at 0x7F35BE2EA730>, 'panoptic_image/filename': '000000000034.png'}
```

Now, convert all raw data to huggingface object detection data format and save to parquet for faster loading. You can do for all variants: 


1. 2017_detection:

```bash
export data_variant="2017_detection"

python scripts/data_processing_scripts/raw_to_parquet_coco.py \
    --coco_dir="data/coco_datasets" \
    --bbox_mode="corners" \
    --data_variant="$data_variant" \
    --save_path="formatted_data/coco_$data_variant"
```

2. 2017_detection_skip:

```bash
export data_variant="2017_detection_skip"

python scripts/data_processing_scripts/raw_to_parquet_coco.py \
    --coco_dir="data/coco_datasets" \
    --bbox_mode="corners" \
    --data_variant="$data_variant" \
    --save_path="formatted_data/coco_$data_variant"
```

<br>

## COCO panoptic segmentation data

COCO panoptic segmentation data is a comprehensive dataset that combines instance segmentation and semantic segmentation tasks. It provides pixel-level annotations for both object instances and scene semantics in complex images. This dataset facilitates the development of models capable of segmenting and understanding images at a detailed level, enabling more sophisticated scene analysis and comprehension.

[Follow these steps at first,](./Vision_Data_Guide.md#downloading-and-formatting-the-coco-dataset-2017) then run the following commands:

You can convert all raw data to huggingface panoptic segmentation data format and save to parquet for faster loading. You can do for all variants:

1. 2017_panoptic:

```bash
export data_variant="2017_panoptic"

python scripts/data_processing_scripts/raw_to_parquet_coco.py \
    --coco_dir="data/coco_datasets" \
    --bbox_mode="corners" \
    --data_variant="$data_variant" \
    --save_path="formatted_data/coco_$data_variant"
```

2. 2017_panoptic_skip:

```bash
export data_variant="2017_panoptic_skip"

python scripts/data_processing_scripts/raw_to_parquet_coco.py \
    --coco_dir="data/coco_datasets" \
    --bbox_mode="corners" \
    --data_variant="$data_variant" \
    --save_path="formatted_data/coco_$data_variant"
```


<br>

## COCO instance segmentation data

COCO instance segmentation data is a dataset specifically designed for the task of segmenting individual objects within images. It includes pixel-level annotations that precisely delineate object boundaries and assign unique identifiers to each instance. This dataset is widely used for training and evaluating models that can accurately segment and differentiate multiple objects in various scenes.

🚧 COCO instance segmentation data downloading and preprocessing steps will be added in future release. Stay tuned!

<br>

## ADE20K semantic segmentation data

ADE20K semantic segmentation data is a dataset focused on semantic segmentation tasks, which involve assigning semantic labels to every pixel in an image. It contains a large collection of images spanning diverse indoor and outdoor scenes, annotated with detailed semantic labels. This dataset enables the training and evaluation of models that can understand and segment images based on their semantic content, making it valuable for applications such as scene understanding and autonomous navigation.

🚧 ADE20K semantic segmentation data downloading and preprocessing steps will be added in future release. Stay tuned!



## Download all processed data at once from s3 bucket

🚧 will be added in future release. Stay tuned!


<!-- If you want to download all our processed data at once without downloading from main websites and process manually,use this commands to download and extract all the formatted data: -->

<!-- 
```bash
mkdir formatted_data
cd formatted_data
wget https://object-detection-anchorblock.s3.ap-south-1.amazonaws.com/data/formatted_data.zip
unzip formatted_data.zip
rm -rf formatted_data.zip
cd ..
```
 -->
