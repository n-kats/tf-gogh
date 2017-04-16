import os

import numpy as np
from PIL import Image
import tensorflow as tf


def load_image(path, size=None):
  """sizeがNoneのときは画像のそのままのサイズで読み込む"""
  img = Image.open(os.path.expanduser(path)).convert("RGB")
  if size is not None:
    img = img.resize(size, Image.BILINEAR)
  return tf.constant(transform_for_train(np.array([np.array(img)[:, :, :3]], dtype=np.float32)))


def transform_for_train(img):
  """
  読み込む画像がRGBなのに対し、VGGなどのパラメータがBGRの順なので、順番を入れ替える。
  ImageNetの色の平均値を引く。
  """
  return img[..., ::-1] - 120


def transform_from_train(img):
  """
  transform_for_trainの逆操作。
  """
  data = img[:, :, ::-1] + 120
  return data.clip(0, 255)
