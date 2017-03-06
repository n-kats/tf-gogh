import tensorflow as tf
import numpy as np

class Conv:

  def __init__(self, chainer_conv):
    W = chainer_conv.W.data
    b = chainer_conv.b.data
    self.W = tf.constant(np.transpose(W, [2, 3, 1, 0]))
    self.b = tf.constant(b)

  def __call__(self, x, pad="VALID", stride=1):
    return tf.nn.conv2d(x, self.W, strides=[1, stride, stride, 1], padding=pad)
