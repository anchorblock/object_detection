# COCO panoptic Labeling Guide

Each pixel in our image have two values associated with it:

[“Z”, “L”] => ["Instance Number", "Label"]

This labeling and prediction format can be expressed as a two channel output, where channel 0 displays each pixel’s label and channel 1 displays each pixels instance.

Typically, 

```segment id = red_pixel + green_pixel * 256 + blue_pixel * 256 * 256```

tensor output/ mask format:

```class_queries_length shape = [batch size, #instance_ids_ (100), #semantic_ids_ (134) ]```

```mask_queries_length shape = [batch size, #instance_ids_ (100), height, width ]```


Have to consider - 
- handle is_crowd option (for is_crowd = 1, we have to merge those instances for that particular image)
- mask label must be in np.int64 format for handling very large digits
- image have to transform (resize, normalize) before passing through label editing scheme/ function.



