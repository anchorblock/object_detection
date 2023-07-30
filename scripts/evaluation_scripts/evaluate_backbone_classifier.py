### INCOMPLETE SCRIPT
### WILL PUBLISH IN NEXT RELEASE


import argparse
import datetime
import sys
sys.setrecursionlimit(10000)
sys.path.append('./')
import argparse
import torch

from PIL import Image
import numpy as np
import math
import os
import json
from datasets import load_dataset, disable_caching
from transformers import (
    AutoConfig, 
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    TrainingArguments,
    Trainer
)
import random
import time
import warnings

from utils.augmentations import generate_transform_function

def speed_up():
    """
    Speeds up the code execution by disabling certain debugging features and setting specific configurations.

    This function performs the following actions:
    1. Disables advisory warnings from the Transformers library.
    2. Disables autograd anomaly detection for PyTorch.
    3. Disables autograd profiling for PyTorch.
    4. Disables emitting NVTX markers for autograd profiling in PyTorch.
    5. Disables emitting ITT markers for autograd profiling in PyTorch.
    6. Sets the CuDNN determinism mode to False for improved performance in CV training tasks.
    7. Disables CuDNN benchmarking mode for improved performance in CV training tasks.

    Note:
    It is important to understand the implications of disabling these debugging features. Disabling
    them might improve performance but can also mask potential issues or decrease debugging capabilities.

    """

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.autograd.profiler.emit_itt(False)
    
    # Training task is CV variable image size; so setting false
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False



def set_seeds(seed = 12345):
    """
    Sets the random seeds for various libraries to ensure reproducibility.

    Args:
        seed (int): The seed value to set for random number generators. Default is 12345.

    Note:
        This function sets the seed value for the following libraries/modules:
        - random: Python's built-in random number generator.
        - os.environ['PYTHONHASHSEED']: Sets the seed for hash-based operations in Python.
        - numpy (np): The NumPy library for numerical computing.
        - torch: The PyTorch library for deep learning.
        - torch.cuda: Sets the seed for CUDA operations in PyTorch.

    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    
    speed_up()
    set_seeds(seed = 42)


    parser = argparse.ArgumentParser(description='Evaluate pretrained model with imagenet-1k validation datasets.')

    parser.add_argument('--pretrained_model_name_or_path', type=str, default = "microsoft/focalnet-tiny", help='Name or path of the pretrained model')
    parser.add_argument('--validation_dataset', type=str, default = "formatted_data/imagenet_1k/validation.parquet", help = "validation datasets (imagenet, parquet format)")
    parser.add_argument('--results_dir', type=str, default = "outputs", help='result directories')

    args = parser.parse_args()


    ### LOAD DATA
    disable_caching()

    imagenet_dataset = load_dataset(
        "parquet", 
        data_files={"validation": args.validation_dataset},
        cache_dir=".cache")
        

    ### LOAD CONFIG, BUILD MODEL AND LOAD PROCESSORS

    model = AutoModelForImageClassification.from_pretrained(args.pretrained_model_name_or_path)
    image_processor = AutoImageProcessor.from_pretrained(args.pretrained_model_name_or_path)


    _transforms = generate_transform_function(
                        image_processor = image_processor,
                        augmentation_config_path="configs/augmentation_config_imagenet.json", 
                        return_mixup_cutmix_fn=False, is_validation = True
                        )


    def transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples
    

    imagenet_dataset = imagenet_dataset.with_transform(transforms)


    data_collator = DefaultDataCollator()


    ## EVALUATION METRICS
    from utils.evaluation import compute_metrics_imagenet1k


    ## EVALUATE
    from torch.utils.data import DataLoader

    dataset_shuffled = imagenet_dataset["validation"].shuffle()

    eval_dataloader = DataLoader(
        dataset_shuffled, 
        batch_size=32, 
        collate_fn=data_collator,
        )
    
    ## EVAL LOOP

    model.eval()

    results_list = []
    print("total_steps: {}".format(len(eval_dataloader)))
    for step, batch in enumerate(eval_dataloader):

        with torch.no_grad():
            outputs = model(**batch)


            results = compute_metrics_imagenet1k(predictions=outputs.logits, references=batch["labels"])
            results_list.append(results)


    import pandas as pd

    flattened_json = pd.json_normalize(results_list)
    df_all = pd.DataFrame.from_dict(flattened_json)
    column_sums = df_all.sum()

    top_1_accuracy = column_sums["topk_accuracy.top1_accuracy_raw"]/column_sums["topk_accuracy.n_samples"]

    top_5_accuracy = column_sums["topk_accuracy.top5_accuracy_raw"]/column_sums["topk_accuracy.n_samples"]

    precision, recall, f1, mAP, n_samples = 0, 0, 0, 0, 0
    for i in range(1000):
        precision += column_sums[f"precision_recall_f1_mAP.{i}.precision_raw"]
        recall += column_sums[f"precision_recall_f1_mAP.{i}.recall_raw"]
        f1 += column_sums[f"precision_recall_f1_mAP.{i}.f1_raw"]
        mAP += column_sums[f"precision_recall_f1_mAP.{i}.mAP_raw"]
        n_samples += column_sums[f"precision_recall_f1_mAP.{i}.n_samples"]

    precision = precision/n_samples
    recall = recall/n_samples
    f1 = f1/n_samples
    mAP = mAP/n_samples


    print(f"top_1_accuracy = {top_1_accuracy}\ntop_5_accuracy = {top_5_accuracy}\nprecision = {precision}\nrecall = {recall}\nf1 = {f1}\nmAP = {mAP}")

    # Create a dictionary to store the evaluation results
    results = {
        "top_1_accuracy": top_1_accuracy,
        "top_5_accuracy": top_5_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP": mAP
    }

    # Save the results to a JSON file
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    with open(f"{args.results_dir}/imagenet_eval_results.json", "w") as file:
        json.dump(results, file, indent=4)

    print("\nEvaluation results saved successfully.")



if __name__ == "__main__":
    main()





