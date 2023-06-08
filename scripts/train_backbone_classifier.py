import sys
sys.path.append('./')

import argparse
import datetime
import sys
sys.setrecursionlimit(10000)
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms

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

import evaluate




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
    7. Enables CuDNN benchmarking mode for improved performance in CV training tasks.

    Note:
    It is important to understand the implications of disabling these debugging features. Disabling
    them might improve performance but can also mask potential issues or decrease debugging capabilities.

    """

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.autograd.profiler.emit_itt(False)
    
    # Training task is CV fixed image size; so setting true to benchmark
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True



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

    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y_%m_%d__%H_%M_%S")

    # Create the argument parser
    parser = argparse.ArgumentParser(description="ImageNet Classifier (backbone) Training")


    parser.add_argument('--model_type', type=str, choices=['bit', 'convnext', 'convnextv2', 'dinat', 'focalnet', 'nat', 'resnet', 'swin'],
                        default='resnet',  # Set the default value to 'resnet'
                        help='Specify the model type (optional if config_path and processor_config_path are given)')

    type_args = parser.parse_args()

    parser.add_argument('--config_path', type=str,
                        default=f'configs/backbones/{type_args.model_type}/config.json',  # Set the default value to 'config.json'
                        help='Specify the path to the model config.json file (optional if model_type is given)')

    parser.add_argument('--processor_config_path', type=str,
                        default=f'configs/backbones/{type_args.model_type}/preprocessor_config.json',  # Set the default value to 'processor_config.json'
                        help='Specify the path to the processor_config.json file (optional if model_type is given)')

    parser.add_argument('--augmentation_config_path', type=str,
                        default='configs/augmentation_config_imagenet.json',
                        help='Specify the path to the augmentation_config.json file')

    parser.add_argument('--train_data_path', type=str, default='formatted_data/imagenet_1k/train.parquet',
                        help='Path to the training data parquet file')
    
    parser.add_argument('--validation_data_path', type=str, default='formatted_data/imagenet_1k/validation.parquet',
                        help='Path to the validation data parquet file')
    
    parser.add_argument('--test_data_path', type=str, default='formatted_data/imagenet_1k/test.parquet',
                        help='Path to the test data parquet file')


    parser.add_argument('--id2label', type=str,
                        default='configs/datasets/imagenet-1k-id2label.json',  # Set the default value to 'configs/datasets/imagenet-1k-id2label.json'
                        help='Specify the path to the id2label.json file (optional)')

    parser.add_argument('--label2id', type=str,
                        default='configs/datasets/imagenet-1k-label2id.json',  # Set the default value to 'configs/datasets/imagenet-1k-label2id.json'
                        help='Specify the path to the label2id.json file (optional)')

    parser.add_argument('--do_mixup_cutmix', type=bool, default=True, help='Specify whether to perform mixup and cutmix')



    args = parser.parse_args()

    ### LOAD DATA
    disable_caching()

    imagenet_dataset = load_dataset(
        "parquet", 
        data_files={"train": args.train_data_path,
                    "validation": args.validation_data_path,
                    "test": args.test_data_path},
        cache_dir=".cache")
        

    # TODO : **************
    ### LOAD CONFIG, BUILD MODEL AND LOAD PROCESSORS

    config = AutoConfig.from_pretrained(args.config_path)

    # read id2label
    with open(args.id2label, 'r') as json_file:
        # Load the JSON data
        id2label = json.load(json_file)

    # read label2id
    with open(args.label2id, 'r') as json_file:
        # Load the JSON data
        label2id = json.load(json_file)

    config.id2label = id2label
    config.label2id = label2id
    config.num_labels = len(label2id.keys())


    model = AutoModelForImageClassification.from_config(config)

    image_processor = AutoImageProcessor.from_pretrained(args.processor_config_path)


    _transforms_train, mixup_cutmix_fn = generate_transform_function(
                        augmentation_config_path=args.augmentation_config_path, 
                        return_mixup_cutmix_fn=True, is_validation = False
                        )
    
    _transforms_valid = generate_transform_function(
                        augmentation_config_path=args.augmentation_config_path, 
                        return_mixup_cutmix_fn=False, is_validation = True
                        )
    

    def transforms_train_data(examples):
        examples["pixel_values"] = [_transforms_train(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples
    
    def transforms_valid_data(examples):
        examples["pixel_values"] = [_transforms_valid(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples
    

    imagenet_dataset["train"] = imagenet_dataset["train"].with_transform(transforms_train_data)
    imagenet_dataset["validation"] = imagenet_dataset["validation"].with_transform(transforms_valid_data)


    data_collator = DefaultDataCollator()


    ## EVALUATION METRICS

    accuracy = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)


    ### TRAIN




    # ```
    # | Parameter                      | Value     |
    # |-------------------------------|----------|
    # | Batch Size                    | 1024     |
    # | Base Learning Rate            | 1e-3     |
    # | Learning Rate Scheduler       | Cosine   |
    # | Minimum Learning Rate         | 1e-5     |
    # | Training Epochs               | 300      |
    # | Warm-up Epochs                | 20       |
    # | Warm-up Schedule              | Linear   |
    # | Warm-up Learning Rate         | 1e-6     |
    # | Optimizer                     | AdamW    |
    # | Stochastic Drop Path Rate     | 0.2      |
    # | Gradient Clip                 | 5.0      |
    # | Weight Decay                  | 0.05     |
    # ```


# custom learning rate schedular likhte hobe (for base and minimum lr)

training_args = TrainingArguments(
    output_dir="my_awesome_food_model",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=food["train"],
    eval_dataset=food["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()






#### Inference


ds = load_dataset("food101", split="validation[:10]")
image = ds["image"][0]


from transformers import pipeline

classifier = pipeline("image-classification", model="my_awesome_food_model") # must pre-loaded id2label, label2id
classifier(image)






































    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Number of warmup steps")
    parser.add_argument("--max_steps", type=int, default=1000000, help="Number of training iterations")
    parser.add_argument("--logging_dir", type=str, default=f"./logs/{date_time_str}", help="Directory for logging")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging interval")
    parser.add_argument("--eval_steps", type=int, default=2000, help="Validate loss every n steps")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save weights every n steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--init_model_directory", type=str, default="./DeBERTaV3", help="Directory for initializing the model")
    parser.add_argument("--save_directory", type=str, default="./DeBERTaV3_trained_bn", help="Directory for saving weights")
    parser.add_argument("--resume_from_checkpoint", type=str, default="./DeBERTaV3", help="Continue path from")
    parser.add_argument("--gradient_checkpointing",type=bool, default=False, help="Enable or disable gradient checkpointing (default: False)")
    
    # Parse the command-line arguments
    args = parser.parse_args()


    # load parquet datasets

    disable_caching() # disabling caching for less storage and speedups

    mlm_datasets = load_dataset("parquet", data_files={"train": args.train_parquet_data_file, "validation": args.test_parquet_data_file})
    
    # load tokenizer, model-config, and model
    tokenizer = DebertaV2TokenizerFast.from_pretrained(args.init_model_directory)
    config = DebertaV2Config.from_pretrained(args.init_model_directory)
    model = DebertaV2ForMaskedLM.from_pretrained(args.init_model_directory)

    # data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


    # training arguments
    training_args = TrainingArguments(
        output_dir = args.save_directory,
        overwrite_output_dir = False,
        evaluation_strategy = "epoch",
        prediction_loss_only = False,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        eval_accumulation_steps = args.gradient_accumulation_steps,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        max_grad_norm = args.max_grad_norm,
        max_steps = args.max_steps,
        lr_scheduler_type = "linear",
        warmup_steps = args.warmup_steps,
        logging_dir = args.logging_dir,
        logging_strategy = "epoch",
        logging_first_step = True,
        logging_steps = args.logging_steps,
        logging_nan_inf_filter = True,
        save_strategy = "epoch",
        save_steps = args.save_steps,
        seed = 42,
        data_seed = 42,
        fp16 = True,
        half_precision_backend="auto", #"cuda_amp",
        tf32 = False,
        local_rank = os.environ["LOCAL_RANK"],
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        eval_steps = args.eval_steps,
        dataloader_num_workers = 2,
        disable_tqdm = False,
        ignore_data_skip = False,
        optim = "adamw_torch",
        report_to = "tensorboard",
        dataloader_pin_memory = True,
        resume_from_checkpoint = args.resume_from_checkpoint,
        gradient_checkpointing = args.gradient_checkpointing,
    )


    loss_func = nn.CrossEntropyLoss()
    

    def compute_metrics(pred): # do only topk 1 while on the fly
        """
        Compute metrics for binary classification.

        Args:
            pred (object): An object containing prediction information.

        Returns:
            dict: A dictionary containing the computed metrics.
                - perplexity (float): The perplexity value computed based on the predictions.
        """
        labels = pred.label_ids
        preds = pred.predictions #.argmax(-1)
        
        labels_tensor = torch.from_numpy(labels).to(torch.float32)
        preds_tensor = torch.from_numpy(preds).to(torch.float32)

        loss = loss_func(preds_tensor.view(-1, config.vocab_size), labels_tensor.view(-1).long())

        try:
            perplexity = math.exp(loss)
        except OverflowError:
            perplexity = float("inf")

        return {"perplexity": perplexity}


    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mlm_datasets["train"],
        eval_dataset=mlm_datasets["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train model
    trainer.train()



    # Save the model
    trainer.save_model(args.save_directory)



if __name__=="__main__":

    main()
    
    print("Training has been completed.")
































































    --per_device_train_batch_size=32 \
    --gradient_accumulation_steps=64 \
    --learning_rate=5e-4 \
    --warmup_steps=10000 \
    --max_steps=1250000 \
    --logging_steps=500 \
    --eval_steps=100000 \
    --save_steps=10000 \
    --init_model_directory="./DeBERTaV3" \
    --save_directory="./DeBERTaV3_trained_bn" \
    --resume_from_checkpoint="./DeBERTaV3" \
    --gradient_checkpointing=true

