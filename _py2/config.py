# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import argparse

import tensorflow as tf


def parse_args():
  parser = argparse.ArgumentParser(description=u'chainer-goghの真似')
  parser.add_argument(u'--model', u'-m', help=u'nin, vggのどちらか')
  parser.add_argument(u'--original_image', u'-i', help=u"[必須]入力の画像", required=True)
  parser.add_argument(u'--style_image', u'-s', help=u"[必須]画風の画像", required=True)
  parser.add_argument(u'--output_dir', u'-o', help=u"生成画像の保存先")
  parser.add_argument(u'--iteration', type=int, help=u"学習回数")
  parser.add_argument(u'--lr', type=float, help=u"学習レート")
  parser.add_argument(u'--lam', type=float, help=u"入力と画風のバランス")
  parser.add_argument(u'--width', type=int, help=u"生成画像の横幅")
  parser.add_argument(u'--height', type=int, help=u"生成画像の高さ")
  parser.add_argument(u'--no_resize_style', action=u'store_true', help=u"画風画像をリサイズせずに使う")

  args = parser.parse_args()
  return args


class Config(object):
  batch_size = 1
  iteration = 5000
  lr = 1.0
  lam = 0.05
  width = 300
  height = 300
  output_shape = [batch_size, height, width, 3]
  output_dir = u"_output"
  model = u"nin"
  # model = "vgg"

  no_resize_style = False  # Trueにすると画風画像をリサイズせずに利用する（開始が遅くなる）

  def __init__(self, args):
    self.model = args.model or self.model
    self.original_image = args.original_image
    self.style_image = args.style_image

    self.iteration = args.iteration or self.iteration
    self.lr = args.lr or self.lr
    self.lam = args.lam or self.lam
    self.width = args.width or self.width
    self.height = args.height or self.height
    self.output_shape = [self.batch_size, self.height, self.width, 3]
    self.output_dir = args.output_dir or self.output_dir

    self.save_path = os.path.expanduser(os.path.join(self.output_dir, u"%05d.png"))
    self.optimizer = tf.train.AdamOptimizer(self.lr)
    self.no_resize_style = args.no_resize_style or self.no_resize_style
