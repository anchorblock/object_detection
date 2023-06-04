# Download Imagenet-1k datasets

# Create a new folder called 'imagenet_1k' inside 'data' and change directory to the 'imagenet_1k' folder
cd data
mkdir imagenet_1k
cd imagenet_1k

axel -aN -n 100 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
# Use 'axel' command-line tool to download the ILSVRC2012 training dataset.
# '-aN' option: Enable multiple connections for faster downloads.
# '-n 100' option: Download with 100 connections
# The dataset is downloaded from the specified URL.


echo "train data download completed ..."
axel -aN -n 100 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

echo "validation data download completed ..."
axel -aN -n 100 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar

echo "test data download completed ..."

# back to root directory
cd ../..