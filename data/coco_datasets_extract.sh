cd data/coco_datasets

# Unzipping the train2017.zip file
unzip train2017.zip
# Removing the original zip file
rm -rf train2017.zip
# Printing a message indicating the completion of extraction
echo "train2017.zip extraction completed ..."

unzip val2017.zip
rm -rf val2017.zip
echo "val2017.zip extraction completed ..."

unzip test2017.zip
rm -rf test2017.zip
echo "test2017.zip extraction completed ..."

unzip annotations_trainval2017.zip
rm -rf annotations_trainval2017.zip
echo "annotations_trainval2017.zip extraction completed ..."

unzip image_info_test2017.zip
rm -rf image_info_test2017.zip
echo "image_info_test2017.zip extraction completed ..."

# back to root directory
cd ../..

