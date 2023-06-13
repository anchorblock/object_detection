from builtins import isinstance
import os
import glob
import json
import logging
import zipfile
import functools
import collections
from PIL import Image
import datasets
import copy


logger = logging.getLogger(__name__)

_VERSION = datasets.Version("1.0.0", "")

_URL = "https://cocodataset.org/#home"

# Copied from https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/object_detection/coco.py
_CITATION = """\
@article{DBLP:journals/corr/LinMBHPRDZ14,
  author    = {Tsung{-}Yi Lin and
               Michael Maire and
               Serge J. Belongie and
               Lubomir D. Bourdev and
               Ross B. Girshick and
               James Hays and
               Pietro Perona and
               Deva Ramanan and
               Piotr Doll{\'{a}}r and
               C. Lawrence Zitnick},
  title     = {Microsoft {COCO:} Common Objects in Context},
  journal   = {CoRR},
  volume    = {abs/1405.0312},
  year      = {2014},
  url       = {http://arxiv.org/abs/1405.0312},
  archivePrefix = {arXiv},
  eprint    = {1405.0312},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/LinMBHPRDZ14},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

# Copied from https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/object_detection/coco.py
_DESCRIPTION = """COCO is a large-scale object detection, segmentation, and
captioning dataset.
Note:
 * Some images from the train and validation sets don't have annotations.
 * Coco 2014 and 2017 uses the same images, but different train/val/test splits
 * The test split don't have any annotations (only images).
 * Coco defines 91 classes but the data only uses 80 classes.
 * Panotptic annotations defines defines 200 classes but only uses 133.
"""

# Copied from https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/object_detection/coco.py
_CONFIG_DESCRIPTION = """
This version contains images, bounding boxes and labels for the {year} version.
"""

Split = collections.namedtuple(
    'Split', ['name', 'images', 'annotations', 'annotation_type']
)


class AnnotationType(object):
    """Enum of the annotation format types.
    Splits are annotated with different formats.
    """

    BBOXES = 'bboxes'
    PANOPTIC = 'panoptic'
    NONE = 'none'

class CocoAnnotation(object):
  """Coco annotation helper class."""

  def __init__(self, annotation_path):
    with open(annotation_path, "r") as f:
      data = json.load(f)
    self._data = data

  @property
  def categories(self):
    """Return the category dicts, as sorted in the file."""
    return self._data['categories']

  @property
  def images(self):
    """Return the image dicts, as sorted in the file."""
    return self._data['images']

  def get_annotations(self, img_id):
    """Return all annotations associated with the image id string."""
    raise NotImplementedError  # AnotationType.NONE don't have annotations


class CocoAnnotationBBoxes(CocoAnnotation):
  """Coco annotation helper class."""

  def __init__(self, annotation_path):
    super(CocoAnnotationBBoxes, self).__init__(annotation_path)

    img_id2annotations = collections.defaultdict(list)
    for a in self._data['annotations']:
      img_id2annotations[a['image_id']].append(a)
    self._img_id2annotations = {
        k: list(sorted(v, key=lambda a: a['id']))
        for k, v in img_id2annotations.items()
    }

  def get_annotations(self, img_id):
    """Return all annotations associated with the image id string."""
    # Some images don't have any annotations. Return empty list instead.
    return self._img_id2annotations.get(img_id, [])


class CocoAnnotationPanoptic(CocoAnnotation):
  """Coco annotation helper class."""

  def __init__(self, annotation_path):
    super(CocoAnnotationPanoptic, self).__init__(annotation_path)
    self._img_id2annotations = {
        a['image_id']: a for a in self._data['annotations']
    }

  def get_annotations(self, img_id):
    """Return all annotations associated with the image id string."""
    return self._img_id2annotations[img_id]


ANNOTATION_CLS = {
    AnnotationType.NONE: CocoAnnotation,
    AnnotationType.BBOXES: CocoAnnotationBBoxes,
    AnnotationType.PANOPTIC: CocoAnnotationPanoptic,
}


DETECTION_FEATURE = datasets.Features(
    {
        "image": datasets.Image(),
        "image/filename": datasets.Value("string"),
        "image/id": datasets.Value("int64"),
        "objects": datasets.Sequence(feature=datasets.Features({
            "id": datasets.Value("int64"),
            "area": datasets.Value("float32"),
            "bbox": datasets.Sequence(
                feature=datasets.Value("float32")
            ),
            "is_crowd": datasets.Value("bool"),
            "category_id": datasets.Value("int64"),
            "category_name": datasets.Value("string"),
            "supercategory_id": datasets.Value("int64"),
            "supercategory_name": datasets.Value("string"),
            "is_thing": datasets.Value("bool")
        })),
    }
)

PANOPTIC_FEATURE = datasets.Features(
    {
        "image": datasets.Image(),
        "image/filename": datasets.Value("string"),
        "image/id": datasets.Value("int64"),
        "panoptic_objects": datasets.Sequence(feature=datasets.Features({
            "id": datasets.Value("int64"),
            "area": datasets.Value("float32"),
            "bbox": datasets.Sequence(
                feature=datasets.Value("float32")
            ),
            "is_crowd": datasets.Value("bool"),
            "category_id": datasets.Value("int64"),
            "category_name": datasets.Value("string"),
            "supercategory_id": datasets.Value("int64"),
            "supercategory_name": datasets.Value("string"),
            "is_thing": datasets.Value("bool")
        })),
        "panoptic_image": datasets.Image(),
        "panoptic_image/filename": datasets.Value("string"),
    }
)
# More info could be added, like segmentation (as png mask), captions,
# person key-points, more metadata (original flickr url,...).



# Copied from https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/object_detection/coco.py
class CocoConfig(datasets.BuilderConfig):
  """BuilderConfig for CocoConfig."""

  def __init__(self, features, splits= None, has_panoptic=False, skip_empty_annotations=False, bbox_mode="corners", **kwargs):
    super(CocoConfig, self).__init__(
        **kwargs
    )
    self.features = features
    self.splits = splits
    self.has_panoptic = has_panoptic
    self.skip_empty_annotations = skip_empty_annotations
    self.bbox_mode = bbox_mode


# Copied from https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/object_detection/coco.py
class Coco(datasets.GeneratorBasedBuilder):
    """Base MS Coco dataset."""

    BUILDER_CONFIGS = [
      CocoConfig(
          name='2017_detection',
          features=DETECTION_FEATURE,
          description=_CONFIG_DESCRIPTION.format(year=2017),
          version=_VERSION,
          splits=[
              Split(
                  name=datasets.Split.TRAIN,
                  images='train2017',
                  annotations='annotations_trainval2017',
                  annotation_type=AnnotationType.BBOXES,
              ),
              Split(
                  name=datasets.Split.VALIDATION,
                  images='val2017',
                  annotations='annotations_trainval2017',
                  annotation_type=AnnotationType.BBOXES,
              ),
              Split(
                  name=datasets.Split.TEST,
                  images='test2017',
                  annotations='image_info_test2017',
                  annotation_type=AnnotationType.NONE,
              ),
          ],
      ),
      CocoConfig(
          name='2017_panoptic',
          features=PANOPTIC_FEATURE,
          description=_CONFIG_DESCRIPTION.format(year=2017),
          version=_VERSION,
          has_panoptic=True,
          splits=[
              Split(
                  name=datasets.Split.TRAIN,
                  images='train2017',
                  annotations='panoptic_annotations_trainval2017',
                  annotation_type=AnnotationType.PANOPTIC,
              ),
              Split(
                  name=datasets.Split.VALIDATION,
                  images='val2017',
                  annotations='panoptic_annotations_trainval2017',
                  annotation_type=AnnotationType.PANOPTIC,
              ),
          ],
      ),
      CocoConfig(
          name='2017_detection_skip',
          features=DETECTION_FEATURE,
          description=_CONFIG_DESCRIPTION.format(year=2017),
          version=_VERSION,
          skip_empty_annotations=True,
          splits=[
              Split(
                  name=datasets.Split.TRAIN,
                  images='train2017',
                  annotations='annotations_trainval2017',
                  annotation_type=AnnotationType.BBOXES,
              ),
              Split(
                  name=datasets.Split.VALIDATION,
                  images='val2017',
                  annotations='annotations_trainval2017',
                  annotation_type=AnnotationType.BBOXES,
              ),
              Split(
                  name=datasets.Split.TEST,
                  images='test2017',
                  annotations='image_info_test2017',
                  annotation_type=AnnotationType.NONE,
              ),
          ],
      ),
      CocoConfig(
          name='2017_panoptic_skip',
          features=PANOPTIC_FEATURE,
          description=_CONFIG_DESCRIPTION.format(year=2017),
          version=_VERSION,
          has_panoptic=True,
          skip_empty_annotations=True,
          splits=[
              Split(
                  name=datasets.Split.TRAIN,
                  images='train2017',
                  annotations='panoptic_annotations_trainval2017',
                  annotation_type=AnnotationType.PANOPTIC,
              ),
              Split(
                  name=datasets.Split.VALIDATION,
                  images='val2017',
                  annotations='panoptic_annotations_trainval2017',
                  annotation_type=AnnotationType.PANOPTIC,
              ),
          ],
      ),
  ]

    DEFAULT_CONFIG_NAME = "2017_panoptic_skip"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            supervised_keys=None,  # Probably needs to be fixed.
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):

        data_dir = self.config.data_dir
        if not data_dir:
            raise ValueError(
                "This script is supposed to work with local (downloaded) COCO 2017 dataset and extracted by ```data/coco_datasets_extract.sh```. The argument `data_dir` in `load_dataset()` is required."
            )

        splits = []
        for split in self.config.splits:
            image_dir = os.path.join(data_dir, split.images)
            annotations_dir = os.path.join(data_dir, "annotations")

            if split.name != "test":
                if self.config.has_panoptic:
                    panoptic_dir = os.path.join(
                            data_dir, 'panoptic_{}'.format(split.images)
                        )
                else:
                   panoptic_dir = None
            else:
                panoptic_dir = None


            splits.append(
                datasets.SplitGenerator(
                    name=split.name, 
                    gen_kwargs={
                        'image_dir': image_dir,
                        'annotation_dir': annotations_dir,
                        'split_name': split.images,
                        'annotation_type': split.annotation_type,
                        'panoptic_dir': panoptic_dir,
                    }
                )
            )
        return splits


    def _generate_examples(self, image_dir, annotation_dir, split_name, annotation_type, panoptic_dir):
        """Generate examples as dicts.
        Args:
        image_dir: `str`, directory containing the images
        annotation_dir: `str`, directory containing annotations
        split_name: `str`, <split_name><year> (ex: train2017, val2017)
        annotation_type: `AnnotationType`, the annotation format (NONE, BBOXES,
            PANOPTIC)
        panoptic_dir: If annotation_type is PANOPTIC, contains the panoptic image
            directory
        Yields:
        example key and data
        """

        if annotation_type == AnnotationType.BBOXES:
            instance_filename = 'instances_{}.json'
        elif annotation_type == AnnotationType.PANOPTIC:
            instance_filename = 'panoptic_{}.json'
        elif annotation_type == AnnotationType.NONE:  # No annotation for test sets
            instance_filename = 'image_info_{}.json'

        skip_empty_annotations = self.config.skip_empty_annotations
        
        # Load the annotations (label names, images metadata,...)
        instance_path = os.path.join(
            annotation_dir,
            instance_filename.format(split_name),
        )
        coco_annotation = ANNOTATION_CLS[annotation_type](instance_path)
        # Each image is a dict:
        # {
        #     'id': 262145,
        #     'file_name': 'COCO_train2017_000000262145.jpg'
        #     'flickr_url': 'http://farm8.staticflickr.com/7187/xyz.jpg',
        #     'coco_url': 'http://images.cocodataset.org/train2017/xyz.jpg',
        #     'license': 2,
        #     'date_captured': '2013-11-20 02:07:55',
        #     'height': 427,
        #     'width': 640,
        # }
        images = coco_annotation.images

        # ClassLabel names should also contains 'id' and
        # and 'supercategory' (in addition to 'name')

        categories_data = coco_annotation.categories

        # Warning: As Coco only use 80 out of the 91 labels, the c['id'] and
        # dataset names ids won't match.
        # So, converting categories dictionary to assign new_id (as many ids are missing serially) and supercategory_id

        supercategory_ids = {}

        for category in categories_data:
            supercategory = category["supercategory"]
            if supercategory not in supercategory_ids:
                supercategory_ids[supercategory] = len(supercategory_ids)

        # Add unique supercategory IDs and new sequential IDs to the categories
        new_categories = copy.deepcopy(categories_data)

        for new_id, category in enumerate(new_categories):
            supercategory = category["supercategory"]
            category["supercategory_id"] = supercategory_ids[supercategory]
            category["new_id"] = new_id

        # change dictionary format: add id as keys
        categories_dict = {}
        for category in new_categories:
            category_id = category["id"]
            categories_dict[category_id] = category


        if self.config.has_panoptic:
            objects_key = 'panoptic_objects'
        else:
            objects_key = 'objects'
        # self.info.features[objects_key]['label'].names = [
        #     c['name'] for c in categories
        # ]
        

        # Iterate over all images
        annotation_skipped = 0
        idx = 0
        for image_info in sorted(images, key=lambda x: x['id']):
            if annotation_type == AnnotationType.BBOXES:
                # Each instance annotation is a dict:
                # {
                #     'iscrowd': 0,
                #     'bbox': [116.95, 305.86, 285.3, 266.03],
                #     'image_id': 480023,
                #     'segmentation': [[312.29, 562.89, 402.25, ...]],
                #     'category_id': 58,
                #     'area': 54652.9556,
                #     'id': 86,
                # }
                instances = coco_annotation.get_annotations(img_id=image_info['id'])
            elif annotation_type == AnnotationType.PANOPTIC:
                # Each panoptic annotation is a dict:
                # {
                #     'file_name': '000000037777.png',
                #     'image_id': 37777,
                #     'segments_info': [
                #         {
                #             'area': 353,
                #             'category_id': 52,
                #             'iscrowd': 0,
                #             'id': 6202563,
                #             'bbox': [221, 179, 37, 27],
                #         },
                #         ...
                #     ]
                # }
                panoptic_annotation = coco_annotation.get_annotations(
                    img_id=image_info['id']
                )
                instances = panoptic_annotation['segments_info']
            else:
                instances = []  # No annotations

            if not instances:
                annotation_skipped += 1
                if skip_empty_annotations:
                    continue

            def build_bbox(x, y, width, height):
                # pylint: disable=cell-var-from-loop
                # build_bbox is only used within the loop so it is ok to use image_info
                return [
                    x,
                    y,
                    x + width,
                    y + height,
                ]
                # pylint: enable=cell-var-from-loop

            example = {
                'image': Image.open(os.path.join(image_dir, image_info['file_name'])),
                'image/filename': image_info['file_name'],
                'image/id': image_info['id'],
                objects_key: [
                    {
                        'id': instance['id'],
                        'area': instance['area'],
                        'bbox': build_bbox(*instance['bbox']) if self.config.bbox_mode == "corners" else instance['bbox'],
                        'is_crowd': bool(instance['iscrowd']),
                        'category_id': categories_dict[instance['category_id']]['id'],
                        'category_name': categories_dict[instance['category_id']]['name'],
                        'supercategory_id': categories_dict[instance['category_id']]['supercategory_id'],
                        'supercategory_name': categories_dict[instance['category_id']]['supercategory'],
                        'is_thing': bool(categories_dict[instance['category_id']]['isthing']) if self.config.has_panoptic else bool(1) 
                    }
                    for instance in instances
                ],
            }
            if self.config.has_panoptic and panoptic_dir is not None:
                panoptic_filename = panoptic_annotation['file_name']
                example['panoptic_image'] = Image.open(os.path.join(panoptic_dir, panoptic_filename))
                example['panoptic_image/filename'] = panoptic_filename

            yield idx, example
            idx += 1

        logging.info(
            '%d/%d images do not contains any annotations',
            annotation_skipped,
            len(images),
        )


