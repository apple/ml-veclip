from typing import List
import numpy as np
import torch

from transformers.utils import is_tf_available

if is_tf_available():
    import tensorflow as tf  # type: ignore
else:
    raise ValueError("Please run `pip install tensorflow` to use the processor.")

MEAN_RGB = [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
STDDEV_RGB = [0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]


def crop_image(image: tf.Tensor, center_crop_fraction: float = 0.875):
    image_size = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    crop_size = center_crop_fraction * tf.math.minimum(image_size[0], image_size[1])
    crop_offset = tf.cast((image_size - crop_size) / 2.0, dtype=tf.int32)
    crop_size = tf.cast(crop_size, dtype=tf.int32)
    return image[
        crop_offset[0] : crop_offset[0] + crop_size, crop_offset[1] : crop_offset[1] + crop_size, :  # noqa: E203
    ]


def whiten(
    image: tf.Tensor,
) -> tf.Tensor:
    image = tf.cast(tf.convert_to_tensor(image), tf.float32)
    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    return image


def tf_image_reshape_crop(image: tf.Tensor, crop_size: int) -> tf.Tensor:
    # 100000 is chosen as no image would have 100000 pixels along one edge.
    shape_1 = (100000, crop_size)
    shape_2 = (crop_size, 100000)
    image = tf.cond(
        tf.shape(image)[0] > tf.shape(image)[1],
        lambda: tf.image.resize(image, shape_1, method="bilinear", preserve_aspect_ratio=True, antialias=False),
        lambda: tf.image.resize(image, shape_2, method="bilinear", preserve_aspect_ratio=True, antialias=False),
    )
    processed_image = crop_image(image=image, center_crop_fraction=1)
    return processed_image


def _single_image_preprocess(image: np.ndarray, crop_size: int = 224, resize_only: bool = False):
    """Single image preprocess.
    Args:
        images: image in numpy array.
        crop_size: the size of the cropped images.
        resize_only: If true, only resize to the crop size, otherwise, first resize then center crop.
    Returns:
        A torch tensor with processed image.
    """
    image = tf.constant(image)
    if resize_only:
        image = tf.image.resize(
            image, (crop_size, crop_size), method="bilinear", preserve_aspect_ratio=False, antialias=False
        )
    else:
        image = tf_image_reshape_crop(image, crop_size)
    image = whiten(image)
    return torch.asarray(image.numpy())


def image_preprocess(images: List[np.ndarray], crop_size: int = 224, resize_only: bool = False):
    """Image preprocess using tf resizing function.
    Args:
        images: A list of numpy array.
        crop_size: the size of the cropped images.
    Returns:
        A torch tensor with shape [size_of_images, crop_size, crop_size, 3].
    """
    processed_images = []
    for image in images:
        image = tf.constant(image)
        processed_image = _single_image_preprocess(image, crop_size=crop_size, resize_only=resize_only)
        processed_images.append(processed_image)
    return torch.permute(torch.stack(processed_images, 0), (0, 3, 1, 2))