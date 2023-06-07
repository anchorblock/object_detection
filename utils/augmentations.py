import torch
from torchvision.transforms import (
    ColorJitter,
    RandomErasing,
    Normalize,
    Compose,
    PILToTensor,
    ConvertImageDtype
)

from timm.data.auto_augment import rand_augment_transform as AutoAugment
from timm.data.mixup import Mixup
import json
from types import SimpleNamespace



def generate_transform_function(image_processor, augmentation_config_path, return_mixup_cutmix_fn = False):
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

    # | Color Jitter Factor           | 0.4      | 
    # | Auto-augmentation             | rand-m9-mstd0.5-inc1 | 
    # | Random Erasing Probability    | 0.25     | 
    # | Random Erasing Mode           | Pixel    | 
    # | Mixup α                       | 0.8      |
    # | Cutmix α                      | 0.8      |
    # | Mixup Probability             | 1.0      |
    # | Mixup Switch Probability      | 0.5      |
    # | Stochastic Drop Path Rate     | 0.2/0.3/0.5 |
    # | Label Smoothing               | 0.1      |

    """

    with open(augmentation_config_path, 'r') as json_file:
        # Load the JSON data
        config = json.load(json_file)
        config = SimpleNamespace(**config)
    
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
        color_jitter,
        auto_augment,
        PILToTensor(),
        ConvertImageDtype(torch.float32),
        normalize,
        random_erasing
        ])


    if return_mixup_cutmix_fn:
        return _transforms, mixup_cutmix_fn
    
    return _transforms


if __name__ == "__main__":

    from PIL import Image
    import requests
    from io import BytesIO
    import matplotlib.pyplot as plt
    from transformers import AutoImageProcessor

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
