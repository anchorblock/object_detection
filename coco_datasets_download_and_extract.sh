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