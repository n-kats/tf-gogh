import tensorflow as tf
import numpy as np
from PIL import Image

from caffe_to_tf import load_caffemodel
from data import transform_from_train


def pool(x, ksize, stride, padding="SAME"):
  return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1],
                        strides=[1, stride, stride, 1],
                        padding=padding)


class BaseModel:
  """
  特徴量を得るためのモデルのAbstract class
  """
  default_caffemodel = None
  default_alpha = None
  default_beta = None

  def __init__(self, caffemodel=None, alpha=None, beta=None):
    self.conv = load_caffemodel(caffemodel or self.default_caffemodel)
    self.alpha = alpha or self.default_alpha
    self.beta = beta or self.default_beta


class NIN(BaseModel):
  """
  NINを用いた特徴量
  """
  default_caffemodel = "nin_imagenet.caffemodel"
  default_alpha = [0., 0., 1., 1.]
  default_beta = [1., 1., 1., 1.]

  def __call__(self, x):
    """NINの特徴量"""
    x0 = self.conv("conv1")(x, stride=4)

    y1 = self.conv("cccp2")(self.conv("cccp1")(x0), activation_fn=None)
    pool1 = pool(tf.nn.relu(y1), ksize=3, stride=2)
    x1 = self.conv("conv2")(pool1, stride=1)

    y2 = self.conv("cccp4")(self.conv("cccp3")(x1), activation_fn=None)
    pool2 = pool(tf.nn.relu(y2), ksize=3, stride=2)
    x2 = self.conv("conv3")(pool2, stride=1)

    y3 = self.conv("cccp6")(self.conv("cccp5")(x2), activation_fn=None)
    pool3 = pool(tf.nn.relu(y3), ksize=3, stride=2)

    drop = tf.nn.dropout(pool3, 0.5)
    x3 = self.conv("conv4-1024")(drop)

    return [x0, x1, x2, x3]


class VGG(BaseModel):
  """
  VGGを用いた特徴量
  """
  default_caffemodel = "VGG_ILSVRC_16_layers.caffemodel"
  default_alpha = [0., 0., 1., 1.]
  default_beta = [1., 1., 1., 1.]

  def __call__(self, x):
    """VGGの特徴量"""
    y1 = self.conv("conv1_2")(self.conv("conv1_1")(x), activation_fn=None)
    x1 = pool(tf.nn.relu(y1), ksize=2, stride=2)  # max?

    y2 = self.conv("conv2_2")(self.conv("conv2_1")(x1), activation_fn=None)
    x2 = pool(tf.nn.relu(y2), ksize=2, stride=2)  # max?

    y3 = self.conv("conv3_3")(self.conv("conv3_2")(self.conv("conv3_1")(x2)), activation_fn=None)
    x3 = pool(tf.nn.relu(y3), ksize=2, stride=2)  # max?

    y4 = self.conv("conv4_3")(self.conv("conv4_2")(self.conv("conv4_1")(x3)), activation_fn=None)

    return [y1, y2, y3, y4]


def generate_model(model_name, **args):
  if model_name == 'nin':
    return NIN(**args)
  if model_name == 'vgg':
    return VGG(**args)


# TODO: よさげにする
def inner_product_matrix(y):
  _, height, width, ch_num = y.get_shape().as_list()
  y_reshaped = tf.reshape(y, [-1, height * width, ch_num])
  return tf.matmul(y_reshaped, y_reshaped, adjoint_a=True) / (height * width * ch_num)
# TODO: version依存


class Generator:
  def __init__(self, base_model, img_orig, img_style, config):
    # 特徴抽出を行う
    mids_orig = base_model(img_orig)
    mids_style = base_model(img_style)

    # 損失関数に使うものを作る
    prods_style = [inner_product_matrix(y) for y in mids_style]

    # img_genを初期化する
    img_gen = tf.Variable(tf.random_uniform(config.output_shape, -20, 20))  # rank 4で無くてもいい説

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
    self.total_loss = sum(self.loss)  # tfのを使うべき？
    self.total_train = config.optimizer.minimize(self.total_loss)
    clipped = tf.clip_by_value(self.img_gen, -120., 136.)
    self.clip = tf.assign(self.img_gen, clipped)

  def generate(self, config):
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      print("start")
      # 学習開始
      for i in range(config.iteration):
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
