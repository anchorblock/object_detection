# Download COCO datasets

mkdir coco_datasets
cd coco_datasets

axel -aN -n 100 http://images.cocodataset.org/zips/train2017.zip
echo "train2017.zip download completed ..."

axel -aN -n 100 http://images.cocodataset.org/zips/val2017.zip
echo "val2017.zip download completed ..."

axel -aN -n 100 http://images.cocodataset.org/zips/test2017.zip
echo "test2017.zip download completed ..."

axel -aN -n 100 http://images.cocodataset.org/annotations/annotations_trainval2017.zip
echo "annotations_trainval2017.zip download completed ..."

axel -aN -n 100 http://images.cocodataset.org/annotations/image_info_test2017.zip
echo "image_info_test2017.zip download completed ..."

cd ..