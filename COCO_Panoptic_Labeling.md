# COCO panoptic Labeling Guide

Each pixel in our image have two values associated with it:

[“L”, “Z”] => ["Label", "Instance Number"]

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*lYd6VtvhWKr1KzJFZpD56g.png" width="550px">

This labeling and prediction format can be expressed as a two channel output, where channel 1 displays each pixel’s label and channel 2 displays each pixels instance.

tensor output/ mask format:

class_queries_length shape = [batch size, #instance_ids_ (100), #semantic_ids_ (134) ]

mask_queries_length shape = [batch size, #instance_ids_ (100), height, width ]