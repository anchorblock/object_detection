import sys
sys.setrecursionlimit(10000)
sys.path.append('./')

import argparse
import datetime
import torch
import math
from torch.optim.lr_scheduler import LambdaLR

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
from utils.augmentations import generate_transform_function
import evaluate


now = datetime.datetime.now()
date_time_str = now.strftime("%Y_%m_%d__%H_%M_%S")


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



def parse_arguments():
    parser = argparse.ArgumentParser(description='ImageNet Classifier (backbone) Training')

    # Model arguments
    parser.add_argument('--model_type', type=str, choices=['bit', 'convnext', 'convnextv2', 'dinat', 'focalnet', 'nat', 'resnet', 'swin'], default='resnet', help='Specify the model type (optional if config_path and processor_config_path are given)')

    type_args = parser.parse_args()

    parser.add_argument('--config_path', type=str, default=f'configs/backbones/{type_args.model_type}/config.json', help='Specify the path to the model config.json file (optional if model_type is given)')
    parser.add_argument('--processor_config_path', type=str, default=f'configs/backbones/{type_args.model_type}/preprocessor_config.json', help='Specify the path to the processor_config.json file (optional if model_type is given)')
    parser.add_argument('--augmentation_config_path', type=str, default='configs/augmentation/augmentation_config_imagenet.json', help='Path to the augmentation config file')

    # Data paths
    parser.add_argument('--train_data_path', type=str, default='formatted_data/imagenet_1k/train.parquet', help='Path to the training data (parquet file)')
    parser.add_argument('--validation_data_path', type=str, default='formatted_data/imagenet_1k/validation.parquet', help='Path to the validation data (parquet file)')
    parser.add_argument('--test_data_path', type=str, default='formatted_data/imagenet_1k/test.parquet', help='Path to the test data (parquet file)')
    parser.add_argument('--id2label', type=str, default='configs/datasets/imagenet-1k-id2label.json', help='Path to the ID to label mapping file')
    parser.add_argument('--label2id', type=str, default='configs/datasets/imagenet-1k-label2id.json', help='Path to the label to ID mapping file')

    # Training options
    parser.add_argument('--do_mixup_cutmix', type=bool, default=True, help='Whether to perform mixup and cutmix during training')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1024, help='Batch size for training on each device')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1024, help='Batch size for evaluation on each device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--learning_rate_scheduler', type=str, default='cosine', help='Learning rate scheduler')
    parser.add_argument('--minimum_learning_rate', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--training_epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='Number of warmup epochs')
    parser.add_argument('--warmup_schedule', type=str, default='custom_cosine', help='Warmup schedule')
    parser.add_argument('--warmup_learning_rate', type=float, default=1e-6, help='Warmup learning rate')
    parser.add_argument('--logging_dir', type=str, default='./logs/{date_time_str}', help='Directory to save logs')
    parser.add_argument('--optimizer', type=str, default='adamw_torch', help='optimier type')


    parser.add_argument('--stochastic_drop_path_rate', type=float, default=0.2, help='Rate of stochastic drop path')
    parser.add_argument('--gradient_clip', type=float, default=5.0, help='Gradient clipping value')
    parser.add_argument('--save_directory', type=str, default=f"outputs/backbone/{type_args.model_type}", help='Directory to save the trained model')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument('--gradient_checkpointing', type=bool, default=False, help='Whether to use gradient checkpointing')
    parser.add_argument('--fp16', type=bool, default=True, help='Whether to use mixed precision training with FP16')
    parser.add_argument('--tf32', type=bool, default=False, help='Whether to use mixed precision training with TF32')

    args = parser.parse_args()
    return args



def main():
    
    speed_up()
    set_seeds(seed = 42)



    # parser arguments
    args = parse_arguments()


    ### LOAD DATA
    disable_caching()

    imagenet_dataset = load_dataset(
        "parquet", 
        data_files={"train": args.train_data_path,
                    "validation": args.validation_data_path,
                    "test": args.test_data_path},
        cache_dir=".cache")
        

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
    config.drop_path_rate = args.stochastic_drop_path_rate



    model = AutoModelForImageClassification.from_config(config)

    image_processor = AutoImageProcessor.from_pretrained(args.processor_config_path)


    _transforms_train, mixup_cutmix_fn = generate_transform_function(
                        image_processor=image_processor,
                        augmentation_config_path=args.augmentation_config_path, 
                        return_mixup_cutmix_fn=True, is_validation = False
                        )
    
    _transforms_valid = generate_transform_function(
                        image_processor=image_processor,
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


    class BackboneTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)


        def compute_loss(self, model, inputs, return_outputs=False):

            # mixup cutmix
            if args.do_mixup_cutmix:
                inputs["pixel_values"], inputs["labels"] = mixup_cutmix_fn(inputs["pixel_values"], inputs["labels"])

            # forward pass
            outputs = model(**inputs)

            # compute custom loss
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss


        def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):

            def lr_lambda(current_step):
                if current_step < self.args.get_warmup_steps(num_training_steps):
                    # Linear warmup phase
                    return (self.args.learning_rate - args.warmup_learning_rate) / self.args.get_warmup_steps(num_training_steps) * current_step + args.warmup_learning_rate
                else:
                    # Cosine annealing phase
                    return args.minimum_learning_rate + 0.5 * (self.args.learning_rate - args.minimum_learning_rate) * (1 + math.cos(math.pi * (current_step - self.args.get_warmup_steps(num_training_steps)) / (num_training_steps - self.args.get_warmup_steps(num_training_steps))))
            
            scheduler = LambdaLR(
                optimizer=self.optimizer if optimizer is None else optimizer,
                lr_lambda=lr_lambda
            )

            self.lr_scheduler = scheduler

            return self.lr_scheduler


    # all params for epochs
    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)

    training_args = TrainingArguments(
        output_dir = args.save_directory,
        overwrite_output_dir = False,
        evaluation_strategy = "epoch",
        prediction_loss_only = False,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size = args.per_device_eval_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        eval_accumulation_steps = args.gradient_accumulation_steps,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        max_grad_norm = args.gradient_clip,
        lr_scheduler_type = args.learning_rate_scheduler,
        num_train_epochs = args.training_epochs,
        warmup_ratio = args.warmup_epochs / args.training_epochs,
        logging_dir = args.logging_dir,
        logging_strategy = "epoch",
        logging_first_step = True,
        logging_nan_inf_filter = True,
        save_strategy = "epoch",
        save_total_limit = 10,
        seed = 42,
        data_seed = 42,
        fp16 = args.fp16,
        half_precision_backend="auto",
        tf32 = args.tf32,
        local_rank = os.environ["LOCAL_RANK"],
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        dataloader_num_workers = 2,
        disable_tqdm = False,
        ignore_data_skip = False,
        optim = "adamw_torch",
        report_to = "tensorboard",
        dataloader_pin_memory = True,
        resume_from_checkpoint = args.resume_from_checkpoint,
        gradient_checkpointing = args.gradient_checkpointing,
    )


    # trainer

    trainer = BackboneTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=imagenet_dataset["train"],
        eval_dataset=imagenet_dataset["validation"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )


    # Train model
    trainer.train()


    # Save the model
    trainer.save_model(args.save_directory)
    image_processor.save_pretrained(args.save_directory)



if __name__=="__main__":

    main()
    
    print("Training has been completed.")

