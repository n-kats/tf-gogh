import os

import numpy as np
from PIL import Image
import tensorflow as tf


def transform_for_train(img):
  return img[..., ::-1] - 120


def load_image(path, size=None):
  img = Image.open(os.path.expanduser(path)).convert("RGB")
  if size is not None:
    img = img.resize(size, Image.BILINEAR)
  return tf.constant(transform_for_train(np.array([np.array(img)[:, :, :3]], dtype=np.float32)))
