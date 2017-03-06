import tensorflow as tf
import numpy as np
from PIL import Image
from chainer.functions import caffe

from caffe_to_tf import Conv


def generate_model(model_name, **args):
  if model_name == 'nin':
    return NIN(**args)
  if model_name == 'vgg':
    return VGG(**args)


class BaseModel:
  """
  特徴量を得るためもモデルのアブストラクトクラス
  """
  default_caffemodel = None
  default_alpha = None
  default_beta = None

  def __init__(self, caffemodel=None,
               alpha=None, beta=None):
    assert self.default_caffemodel is not None
    if caffemodel is None:
      caffemodel = self.default_caffemodel

    assert self.default_alpha is not None
    if alpha is None:
      alpha = self.default_alpha

    assert self.default_beta is not None
    if beta is None:
      beta = self.default_beta

    print("load model... %s" % caffemodel)
    self.model = caffe.CaffeFunction(caffemodel)
    self.alpha = alpha
    self.beta = beta


class NIN(BaseModel):
  """
  NINを用いた特徴量
  """
  default_caffemodel = "nin_imagenet.caffemodel"
  default_alpha = [0., 0., 1., 1.]
  default_beta = [1., 1., 1., 1.]

  def __call__(self, x):
    y0 = tf.nn.relu(Conv(self.model.conv1)(x, pad="VALID", stride=4))
    y1 = Conv(self.model.cccp2)(
              tf.nn.relu(Conv(self.model.cccp1)(y0)))
    pool0 = tf.nn.avg_pool(tf.nn.relu(y1), ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding="VALID")
    x1 = tf.nn.relu(Conv(self.model.conv2)(pool0, stride=1, pad="SAME"))
    y2 = Conv(self.model.cccp4)(
              tf.nn.relu(Conv(self.model.cccp3)(x1)))
    pool2 = tf.nn.avg_pool(tf.nn.relu(y2), ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding="VALID")
    x2 = tf.nn.relu(Conv(self.model.conv3)(pool2, pad="SAME"))
    y3 = Conv(self.model.cccp6)(
              tf.nn.relu(Conv(self.model.cccp5)(x2)))
    pool3 = tf.nn.avg_pool(tf.nn.relu(y3), ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding="VALID")
    drop = tf.nn.dropout(pool3, 0.5)
    x3 = tf.nn.relu(Conv(getattr(self.model, "conv4-1024"))(drop, pad="SAME"))
    return [y0, x1, x2, x3]


class VGG(BaseModel):
  """
  VGGを用いた特徴量
  """
  default_caffemodel = "VGG_ILSVRC_16_layers.caffemodel"
  default_alpha = [0, 0, 1, 1]
  default_beta = [1, 1, 1, 1]

  def __call__(self, x):
    y1 = Conv(self.model.conv1_2)(
        tf.nn.relu(Conv(self.model.conv1_1)(x)))
    x1 = tf.nn.avg_pool(tf.nn.relu(y1), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    y2 = Conv(self.model.conv2_2)(
        tf.nn.relu(Conv(self.model.conv2_1)(x1)))
    x2 = tf.nn.avg_pool(tf.nn.relu(y2), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    y3 = Conv(self.model.conv3_3)(
        tf.nn.relu(Conv(self.model.conv3_2)(
                   tf.nn.relu(Conv(self.model.conv3_1)(x2)))))
    x3 = tf.nn.avg_pool(tf.nn.relu(y3), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    y4 = Conv(self.model.conv4_3)(
        tf.nn.relu(Conv(self.model.conv4_2)(
                   tf.nn.relu(Conv(self.model.conv4_1)(x3)))))
    return [y1, y2, y3, y4]


def inner_product_matrix(y):
  _, height, width, ch_num = y.get_shape().as_list()
  y_reshaped = tf.reshape(y, [-1, height * width, ch_num])
  return tf.matmul(y_reshaped, y_reshaped, adjoint_a=True) / (height * width * ch_num)


class Generator:
  def __init__(self, base_model, img_orig, img_style, config):
    # 特徴抽出を行う
    mids_orig = base_model(img_orig)
    mids_style = base_model(img_style)

    # 損失関数に使うものを作る
    prods_style = [inner_product_matrix(y) for y in mids_style]

    # img_genを初期化する
    img_gen = tf.Variable(tf.random_uniform(config.output_shape, -20, 20))

    self.img_gen = img_gen
    mids = base_model(img_gen)

    self.loss = []
    self.loss1 = []
    self.loss2 = []

    for i, (mid, mid_orig, mid_style, prod_style, alpha, beta) in enumerate(
        zip(mids, mids_orig, mids_style, prods_style, base_model.alpha, base_model.beta)):
      # 損失関数の定義
      shape1 = mid.get_shape().as_list()
      loss1 = config.lam * tf.nn.l2_loss(mid - mid_orig) / np.prod(shape1)

      shape2 = prod_style.get_shape().as_list()
      loss2 = beta * tf.nn.l2_loss(inner_product_matrix(mid) - prod_style) / np.prod(shape2)
      if alpha != 0.0:
        loss = loss1 * alpha + loss2 / len(mids)
      else:
        loss = loss2 / len(mids)
      self.loss.append(loss)
      self.loss1.append(loss1 * alpha)
      self.loss2.append(loss2 / len(mids))

    self.total_loss = sum(self.loss)
    self.total_train = config.optimizer.minimize(self.total_loss)
    clipped = tf.clip_by_value(self.img_gen, -120., 136.)
    self.clip = tf.assign(self.img_gen, clipped)

  def generate(self, config):
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      print("start")

      # 学習開始
      for i in range(config.iter):
        sess.run([self.total_train, self.clip])
        if (i + 1) % 50 == 0:
          # l, l1, l2 = sess.run([self.loss, self.loss1, self.loss2])
          # print("%d| loss: %f, loss1: %f, loss2: %f" % (i + 1, sum(l), sum(l1), sum(l2)))
          # for l_, l1_, l2_ in zip(l, l1, l2):
          #   print("\tloss: %f, loss1: %f, loss2: %f" % (l_, l1_, l2_))
          self.save_image(sess, config.save_path % (i + 1))

  def save_image(self, sess, path):
    data = sess.run(self.img_gen)[0]
    data = transform_from_train(data)

    img = Image.fromarray(data.astype(np.uint8))
    print("save %s" % path)
    img.save(path)


def transform_from_train(img):
  data = img[:, :, ::-1] + 120
  return data.clip(0, 255)
