import argparse
import os
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='raw_to_parquet_imagenet.py')
    parser.add_argument('--imagenet_dir', type=str, default = "data/imagenet_1k", help='Path to the ImageNet dataset directory')
    parser.add_argument('--save_path', type=str, default='formatted_data/imagenet_1k', help='Path to save the formatted data (default: formatted_data/imagenet_1k)')

    args = parser.parse_args()

    imagenet_dataset = load_dataset("utils/imagenet_1k_dataset_script.py", data_dir=args.imagenet_dir, splits = ["train", "validation", "test"], cache_dir=".cache")


    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path) 


    imagenet_dataset["train"].to_parquet(f"{args.save_path}/train.parquet") 
    print(f"{args.save_path}/train.parquet has been saved. ")

    imagenet_dataset["validation"].to_parquet(f"{args.save_path}/validation.parquet") 
    print(f"{args.save_path}/validation.parquet has been saved. ")

    imagenet_dataset["test"].to_parquet(f"{args.save_path}/test.parquet") 
    print(f"{args.save_path}/test.parquet has been saved. ")
