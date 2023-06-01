# Download Imagenet-1k datasets

cd data
mkdir imagenet_1k
cd imagenet_1k

axel -aN -n 100 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

echo "train data download completed ..."
axel -aN -n 100 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

echo "validation data download completed ..."
axel -aN -n 100 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar

echo "test data download completed ..."
cd ../..