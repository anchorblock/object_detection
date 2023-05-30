# object_detection

Object detection benchmarking with coco dataset using different transformers backbones and different architectures for panoptic segmentation task. We are using tiny version of every models and every backbones for training and inference convenience.

## To-Do

Releases and To-Do archive lists, referring to [TO_DO.md](./TO_DO.md)


## Install Requirements

To setup installation, referring to [INSTALL.md](./INSTALL.md)

## Download ImageNet dataset and format for training

For using the ImageNet-1k dataset (2012), you need to download it manually first.

```bash
bash imagenet_1k_download_and_extract.sh
```

For testing purpose, you can use the hosted dataset as follows:

```python
import datasets

IMAGENET_DIR = "imagenet_1k"
ds = datasets.load_dataset("utils/imagenet_1k_dataset_script.py", "2017", data_dir=IMAGENET_DIR)
ds["train"][0]
```


## Download COCO dataset and format for training

For using the COCO dataset (2017), you need to download it manually first:

```bash
mkdir coco_datasets
cd coco_datasets
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
unzip annotations_trainval2017.zip
unzip image_info_test2017.zip
```

```bash
cd ..
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
