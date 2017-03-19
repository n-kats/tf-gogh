import os

import tensorflow as tf


class Config:
  batch_size = 1

  def __init__(self, args):
    self.iter = args.iter
    self.lr = args.lr
    self.lam = args.lam
    self.output_shape = [self.batch_size, args.height, args.width, 3]
    self.save_path = os.path.expanduser(os.path.join(args.out_dir, "%05d.png"))
    self.optimizer = tf.train.AdamOptimizer(args.lr)
    self.no_resize_style = args.no_resize_style
