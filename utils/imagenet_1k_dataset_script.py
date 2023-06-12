import os
import glob
from typing import Tuple
import datasets
from datasets.builder import BuilderConfig
from datasets.tasks import ImageClassification
from PIL import Image
from .imagenet_1k_classes import IMAGENET2012_CLASSES
from typing import List


_CITATION = """\
@article{imagenet15russakovsky,
    Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = { {ImageNet Large Scale Visual Recognition Challenge} },
    Year = {2015},
    journal   = {International Journal of Computer Vision (IJCV)},
    doi = {10.1007/s11263-015-0816-y},
    volume={115},
    number={3},
    pages={211-252}
}
"""

_HOMEPAGE = "https://image-net.org/index.php"

_DESCRIPTION = """\
ILSVRC 2012, commonly known as 'ImageNet' is an image dataset organized according to the WordNet hierarchy. Each meaningful concept in WordNet, possibly described by multiple words or word phrases, is called a "synonym set" or "synset". There are more than 100,000 synsets in WordNet, majority of them are nouns (80,000+). ImageNet aims to provide on average 1000 images to illustrate each synset. Images of each concept are quality-controlled and human-annotated. In its completion, ImageNet hopes to offer tens of millions of cleanly sorted images for most of the concepts in the WordNet hierarchy. ImageNet 2012 is the most commonly used subset of ImageNet. This dataset spans 1000 object classes and contains 1,281,167 training images, 50,000 validation images and 100,000 test images
"""


class Imagenet1k(datasets.GeneratorBasedBuilder):
    """A dataset script to work with the local (downloaded) ImageNet dataset"""
    
    def __init__(self, splits: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.splits = splits
    
    VERSION = datasets.Version("1.0.0")

    DEFAULT_WRITER_BATCH_SIZE = 1000

    def _info(self):
        assert len(IMAGENET2012_CLASSES) == 1000
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.Value("int64"),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=[ImageClassification(image_column="image", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = self.config.data_dir
        if not data_dir:
            raise ValueError(
                "This script is supposed to work with local (downloaded) imagenet-1k dataset. The argument `data_dir` in `load_dataset()` is required."
            )
        
        train_files = glob.glob(os.path.join(data_dir, "train") + "/**/*.JPEG", recursive=True)
        val_files = glob.glob(os.path.join(data_dir, "val") + "/**/*.JPEG", recursive=True)
        test_files = glob.glob(os.path.join(data_dir, "test") + "/**/*.JPEG", recursive=True)

        splits = []
        for split in self.splits:
            if split == 'train':
                dataset = datasets.SplitGenerator(
                                name=datasets.Split.TRAIN,
                                gen_kwargs={
                                    "filepaths": train_files,
                                    "split": "train",
                                },
                            )
            elif split in ['val', 'valid', 'validation', 'dev']:
                dataset = datasets.SplitGenerator(
                                name=datasets.Split.VALIDATION,
                                gen_kwargs={
                                    "filepaths": val_files,
                                    "split": "validation",
                                },
                            )
            elif split == 'test':
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepaths": test_files,
                        "split": "test",
                    },
                )
            else:
                continue

            splits.append(dataset)

        return splits


    def _generate_examples(self, filepaths, split):
        """Yields examples."""
        idx = 0
        for path in filepaths:
            if path.endswith(".JPEG"):
                if split == "train":
                    # image filepath format: <CLASS_ID>_<FILE_SERIAL>.JPEG
                    (root, filename) = os.path.split(path)
                    split_filename = filename.split('_')
                    class_id = split_filename[0]
                    label = list(IMAGENET2012_CLASSES.keys()).index(class_id)

                if split in ['val', 'valid', 'validation', 'dev']:
                    # image filepath format: <CLASS_ID>/ILSVRC2012_val_<FILE_SERIAL>.JPEG
                    (root, filename) = os.path.split(path)
                    class_id = os.path.split(root)[-1]
                    label = list(IMAGENET2012_CLASSES.keys()).index(class_id)

                else:
                    label = -1
                
                image = Image.open(path)

                ex = {"image": image, "label": label}
                yield idx, ex
                idx += 1

