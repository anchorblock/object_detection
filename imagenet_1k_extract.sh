cd data/imagenet_1k


mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
# Extract training set; remove compressed file
tar -xvf ILSVRC2012_img_train.tar 
rm -rf ILSVRC2012_img_train.tar
#
# At this stage imagenet_1k/train will contain 1000 compressed .tar files, one for each category
#
# For each .tar file: 
#   1. create directory with same name as .tar file
#   2. extract and copy contents of .tar file into directory
#   3. remove .tar file
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
#
# This results in a training directory like so:
#
#  imagenet_1k/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#


# Change back to original directory
cd ..
#
# Extract the validation data and move images to subfolders:
#
# Create validation directory; move .tar file; change directory; 
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val

# extract validation .tar; remove compressed file
tar -xvf ILSVRC2012_img_val.tar
rm -f ILSVRC2012_img_val.tar


# get script from soumith and run; this script creates all class directories and moves images into corresponding directories


wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
#
# This results in a validation directory like so:
#
#  imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
#



# Change back to original directory
cd ..
#
# Extract the test data and move images to subfolders:
#
# Create test directory; move .tar file; change directory; 
mkdir test && mv ILSVRC2012_img_test_v10102019.tar test/ && cd test

# extract validation .tar; remove compressed file
tar -xvf ILSVRC2012_img_test_v10102019.tar
rm -f ILSVRC2012_img_test_v10102019.tar



# Change back to original directory
cd ..


# Check total files after extract
#
find train/ -name "*.JPEG" | wc -l
#  1281167
find val/ -name "*.JPEG" | wc -l
#  50000
find test/ -name "*.JPEG" | wc -l
#  100000

cd ../..
