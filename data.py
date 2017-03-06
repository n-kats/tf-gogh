import numpy as np
from PIL import Image
import tensorflow as tf
import os


def transform_for_train(img):
  img = img.transpose()[::-1].transpose()
  return img - 120


def load_image(path, size=None):
  img = Image.open(os.path.expanduser(path)).convert("RGB")
  if size is not None:
    img = img.resize(size, Image.BILINEAR)
  return tf.constant(transform_for_train(np.array([np.array(img)[:, :, :3]], dtype=np.float32)))
