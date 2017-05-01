import os
import argparse

import tensorflow as tf


def parse_args():
  parser = argparse.ArgumentParser(description='chainer-goghの真似')
  parser.add_argument('--model', '-m', help='nin, vggのどちらか')
  parser.add_argument('--original_image', '-i', help="[必須]入力の画像", required=True)
  parser.add_argument('--style_image', '-s', help="[必須]画風の画像", required=True)
  parser.add_argument('--output_dir', '-o', help="生成画像の保存先")
  parser.add_argument('--iteration', type=int, help="学習回数")
  parser.add_argument('--lr', type=float, help="学習レート")
  parser.add_argument('--lam', type=float, help="入力と画風のバランス")
  parser.add_argument('--width', type=int, help="生成画像の横幅")
  parser.add_argument('--height', type=int, help="生成画像の高さ")
  parser.add_argument('--no_resize_style', action='store_true', help="画風画像をリサイズせずに使う")

  args = parser.parse_args()
  return args


class Config:
  batch_size = 1
  iteration = 5000
  lr = 1.0
  lam = 0.05
  width = 300
  height = 300
  output_shape = [batch_size, height, width, 3]
  output_dir = "_output"
  model = "nin"
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

    self.save_path = os.path.expanduser(os.path.join(self.output_dir, "%05d.png"))
    self.optimizer = tf.train.AdamOptimizer(self.lr)
    self.no_resize_style = args.no_resize_style or self.no_resize_style
