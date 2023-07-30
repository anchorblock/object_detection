import argparse
import os
from datasets import load_dataset

if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser(description='raw_to_parquet_coco.py')

    parser.add_argument('--coco_dir', type=str, default="data/coco_datasets", help='Path to the COCO dataset directory')

    # Add arguments for bbox_mode and data_variant
    parser.add_argument('--bbox_mode', type=str, default="corners", choices=["corners", "height_width"], help='Bounding box mode: "corners" or "height_width"')

    parser.add_argument('--data_variant', type=str, default="2017_panoptic", choices=["2017_detection", "2017_panoptic", "2017_detection_skip", "2017_panoptic_skip"], help='Data variant: "2017_detection", "2017_panoptic", "2017_detection_skip", or "2017_panoptic_skip"')

    parser.add_argument('--save_path', type=str, default=f"", help='Path to save the formatted data (default: formatted_data/coco_<data_variant>)')

    temp_args = parser.parse_args()

    # Set the default value for --save_path if it is not provided
    if not temp_args.save_path:
        temp_args.save_path = f"formatted_data/coco_{temp_args.data_variant}"

    args = parser.parse_args()


    # loading full COCO dataset
    coco_dataset = load_dataset(path = "utils/dataset_utils/coco_dataset_script.py", name = args.data_variant, bbox_mode = args.bbox_mode, data_dir=args.coco_dir, cache_dir=".cache")

    # creating saving directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path) 


    # saving data to parquet formats
    coco_dataset["train"].to_parquet(f"{args.save_path}/train.parquet") 
    print(f"{args.save_path}/train.parquet has been saved. ")

    coco_dataset["validation"].to_parquet(f"{args.save_path}/validation.parquet") 
    print(f"{args.save_path}/validation.parquet has been saved. ")

    if "test" in coco_dataset:
        coco_dataset["test"].to_parquet(f"{args.save_path}/test.parquet") 
        print(f"{args.save_path}/test.parquet has been saved. ")
