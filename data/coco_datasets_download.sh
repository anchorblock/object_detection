# Download COCO datasets

# Create a new folder called 'coco_datasets' inside 'data' and change directory to the 'coco_datasets' folder
cd data
mkdir coco_datasets
cd coco_datasets

axel -aN -n 100 http://images.cocodataset.org/zips/train2017.zip
echo "train2017.zip download completed ..."
# Use 'axel' command-line tool to download the COCO training dataset.
# '-aN' option: Enable multiple connections for faster downloads.
# '-n 100' option: Download with 100 connections
# The dataset is downloaded from the specified URL.

axel -aN -n 100 http://images.cocodataset.org/zips/val2017.zip
echo "val2017.zip download completed ..."

axel -aN -n 100 http://images.cocodataset.org/zips/test2017.zip
echo "test2017.zip download completed ..."

axel -aN -n 100 http://images.cocodataset.org/annotations/annotations_trainval2017.zip
echo "annotations_trainval2017.zip download completed ..."

axel -aN -n 100 http://images.cocodataset.org/annotations/image_info_test2017.zip
echo "image_info_test2017.zip download completed ..."

# back to root directory
cd ../..