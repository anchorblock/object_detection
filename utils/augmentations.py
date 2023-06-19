import torch
from torchvision.transforms import (
    ColorJitter,
    RandomErasing,
    RandomResizedCrop,
    Resize,
    Normalize,
    Compose,
    PILToTensor,
    ToTensor,
    ConvertImageDtype
)

from torchvision.transforms.v2 import ScaleJitter
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from timm.data.auto_augment import rand_augment_transform as AutoAugment
from timm.data.mixup import Mixup
import json
from types import SimpleNamespace

### Utilities
def pil_to_tensor(PIL_image):

    # Convert PIL image to NumPy array
    image_array = np.array(PIL_image) # to extract mask values
    image_array = np.transpose(image_array, (2, 0, 1))  # (H, W, C) to (C, H, W)

    # Convert NumPy array to PyTorch tensor
    tensor_image = torch.as_tensor(image_array, dtype=torch.int32)

    return tensor_image


### IMAGENET-1k TRANSFORMATION FUNCTION
def generate_transform_function(image_processor, augmentation_config_path, return_mixup_cutmix_fn = False, is_validation = False):
    """
    Generate a transformation function for image processing.

    Args:
        image_processor (ImageProcessor): An instance of the ImageProcessor class that contains image processing methods.
        augmentation_config_path (str): The path to the configuration json file for augmentation.
        return_mixup_cutmix_fn (bool, optional): Whether to return the mixup_cutmix_fn along with the transforms.
                                                 Defaults to False.

    Returns:
        transforms (callable): The transformation function that applies image processing and augmentation.
        
    If return_mixup_cutmix_fn is True:
        Returns:
            transforms (callable): The transformation function that applies image processing and augmentation.
            mixup_cutmix_fn (callable): The mixup_cutmix_fn function for applying mixup and cutmix augmentation.

            
    ImageNet Augmentation Details (from focalnet paper):

     | Color Jitter Factor           | 0.4      | 
     | Auto-augmentation             | rand-m9-mstd0.5-inc1 | 
     | Random Erasing Probability    | 0.25     | 
     | Random Erasing Mode           | Pixel    | 
     | Mixup α                       | 0.8      |
     | Cutmix α                      | 0.8      |
     | Mixup Probability             | 1.0      |
     | Mixup Switch Probability      | 0.5      |
     | Label Smoothing               | 0.1      |

    """

    with open(augmentation_config_path, 'r') as json_file:
        # Load the JSON data
        config = json.load(json_file)
        config = SimpleNamespace(**config)
    
    # random_resized_crop
    size = (
        (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    random_resized_crop = RandomResizedCrop(size)
    resize = Resize(size)


    # color jitter
    color_jitter = ColorJitter(brightness=config.color_jitter_factor, 
                                        contrast=config.color_jitter_factor,
                                        saturation=config.color_jitter_factor, 
                                        hue=config.color_jitter_factor)

    # auto augment
    auto_augment = AutoAugment(
        config_str=config.auto_augmentation_policy, 
        hparams={'translate_const': config.translate_const, 'img_mean': tuple(config.PIL_img_mean)})

    # random erasing
    random_erasing = RandomErasing(p=config.random_erasing_probability, value='random', inplace=False)


    # mixup, cutmix

    mixup_args = {
        'mixup_alpha': config.mixup_alpha,
        'cutmix_alpha': config.cutmix_alpha,
        'cutmix_minmax': None,
        'prob': config.mixup_probability,
        'switch_prob': config.mixup_switch_probability,
        'mode': 'elem',
        'label_smoothing': config.label_smoothing,
        'num_classes': config.num_classes
        }

    mixup_cutmix_fn = Mixup(**mixup_args)


    # normalize
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)


    _transforms = Compose([
        random_resized_crop,
        color_jitter,
        auto_augment,
        PILToTensor(),
        ConvertImageDtype(torch.float32),
        normalize,
        random_erasing
        ])

    if is_validation:
        _transforms = Compose([
            resize,
            PILToTensor(),
            ConvertImageDtype(torch.float32),
            normalize
            ])
        
    if return_mixup_cutmix_fn:
        return _transforms, mixup_cutmix_fn
    
    
    return _transforms



### COCO_PANOPTIC TRANSFORMATION FUNCTION
def generate_transform_function_panoptic(image_processor, augmentation_config_path, is_validation = False, is_mask = False):
    """
    Generate a transformation function for image processing.

    Args:
        image_processor (ImageProcessor): An instance of the ImageProcessor class that contains image processing methods.
        augmentation_config_path (str): The path to the configuration json file for augmentation.
        is_validation (bool)
        is_mask (bool)

    Returns:
        transforms (callable): The transformation function that applies image processing and augmentation.
        
    while training: (from mask2former paper)

    | Parameter                     | Value     |
    |-------------------------------|-----------|
    | large-scale jittering (LSJ)   | 0.1-2.0   |
    | fixed size crop               | 1024×1024 |

    while inference: (from mask2former paper)

    | Parameter                     | Value     |
    |-------------------------------|-----------|
    | Resize: shorter side          | upto 800  |
    | Resize: longer side           | upto 1333 |

    """

    with open(augmentation_config_path, 'r') as json_file:
        # Load the JSON data
        config = json.load(json_file)
        config = SimpleNamespace(**config)

    # large-scale jittering (LSJ) then fixed size crop
    lsj_augmentation = ScaleJitter(target_size=(config.fixed_size_crop[0], config.fixed_size_crop[1]), scale_range=(config.large_scale_jittering[0], config.large_scale_jittering[1]))


    # resize
    if is_validation == False: # for training, needs 1024x1024
        size = (config.fixed_size_crop[0], config.fixed_size_crop[1])

    else:
        size = (
                (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])
                if "shortest_edge" in image_processor.size
                else (image_processor.size["height"], image_processor.size["width"])
            )


    resize = Resize(size, interpolation=InterpolationMode.BILINEAR)
    if is_mask == True:
        resize = Resize(size, interpolation=InterpolationMode.NEAREST)



    # normalize
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

    # transforms function
    if is_mask == True:
        _transforms = Compose([
            resize,
            pil_to_tensor,
            ])
    elif is_validation == False: #(train)
        _transforms = Compose([
            lsj_augmentation,
            resize,
            PILToTensor(),
            ConvertImageDtype(torch.float32),
            normalize
            ])
    else: #(validation)
        _transforms = Compose([
            resize,
            PILToTensor(),
            ConvertImageDtype(torch.float32),
            normalize
            ])


    return _transforms



if __name__ == "__main__":

    from PIL import Image
    import requests
    from io import BytesIO
    import matplotlib.pyplot as plt
    from transformers import AutoImageProcessor

    ## IMAGENET AUGMENTATION TEST
    # read image
    IMG_URL = "https://t4.ftcdn.net/jpg/05/68/28/05/360_F_568280532_Bvxwd66M3Y22vVeJ3VRqHRAqrdNfJo7o.jpg"

    response = requests.get(IMG_URL)
    image = Image.open(BytesIO(response.content))

    # load image processor and augmentation config
    image_processor = AutoImageProcessor.from_pretrained("configs/backbones/focalnet/preprocessor_config.json")
    augmentation_config_path = "configs/augmentation_config_imagenet.json"


    transforms_fn = generate_transform_function(image_processor, augmentation_config_path)

    # transform, and show image
    img_aug = transforms_fn(image)
    plt.imshow(img_aug.permute(1, 2, 0))


    ## COCO AUGMENTATION TEST
    # read image
    IMG_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

    response = requests.get(IMG_URL)
    image = Image.open(BytesIO(response.content))

    # load image processor and augmentation config
    image_processor = AutoImageProcessor.from_pretrained("configs/architectures/custom_mask2former/preprocessor_config.json")
    augmentation_config_path = "configs/augmentation/augmentation_config_coco_panoptic.json"


    transforms_fn = generate_transform_function_panoptic(image_processor, augmentation_config_path, is_validation = False, is_mask = False)

    # transform, and show image
    img_aug = transforms_fn(image)
    plt.imshow(img_aug.permute(1, 2, 0))