# %% [code]
"""
Majority of the code comes from here:
https://github.com/tensorflow/tpu/blob/master/models/experimental/inception/inception_preprocessing.py
"""

import tensorflow as tf


def distorted_bounding_box_crop(
    image,
    bbox,
    min_object_covered=0.1,
    aspect_ratio_range=(3.0 / 4.0, 4.0 / 3.0),
    area_range=(0.05, 1.0),
    max_attempts=100,
    scope=None,
):
    """Generates cropped_image using a one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.
    Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding box
        supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    scope: Optional scope for name_scope.
    Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True,
    )
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox


def inception_crop(image, height=128, width=128, bbox=None):
    """Applies "Inception-style cropping" (https://arxiv.org/abs/1409.4842).
    Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
    Returns:
    3-D float Tensor of distorted image used for training with range [-1, 1].
    """
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    distorted_image, distorted_bbox = distorted_bounding_box_crop(image, bbox)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None, 3])

    # Resize the image to the provided shape.
    distorted_image = tf.image.resize(distorted_image, (height, width))
    return distorted_image
