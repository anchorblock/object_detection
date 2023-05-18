# object_detection

Object detection benchmarking with coco dataset using different transformers backbone (maskformer method).

## Install Requirements

To setup installation, referring to [INSTALL.md](./INSTALL.md)

## Download dataset and format for training

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

COCO_DIR = ...(path to the downloaded dataset directory)...
ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR)
```

## Train maskformer model for different backbones

To train maskformer model, firstly different image classification models will be pretrained with imagenet. Then, maskformer model will be built based on pre-trained backbone.












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
