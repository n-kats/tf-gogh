# 機械学習勉強会資料
## 使い方
* このソースコードをダウンロード
* 特徴量のパラメータを次のリンクからcaffemodelをダウンロードし、このファイルと同じディレクトリに配置する(ファイル名を変えないでください)
  * [nin_imagenet.caffemodel](https://gist.github.com/mavenlin/d802a5849de39225bcc6)
  * [VGG_ILSVRC_16_layers.caffemodel](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)
* 必要なライブラリをインストール(pyenvなどを使うべき)
 `pip3 install -U -r requirements.txt`

## 概要
chainer-goghを参考にtensorflowで書いたもの。caffemodelを読み込む部分でchainerのcaffemodelを読み込むものを使っているが、それ以外はtensorflow。

## 利用例（pyenv-virtualenvを使う例）
```
git clone https://github.com/n-kats/tf-gogh.git
cd tf-gogh
pyenv install 3.6.0
pyenv virtualenv 3.6.0 benkyoukai
pyenv local benkyoukai
pip install -U -r requirements.txt
wget https://www.dropbox.com/s/0cidxafrb2wuwxw/nin_imagenet.caffemodel
python main.py \
  -m nin \
  -i 元画像パス \
  -s 画風画像パス
```

VGGの場合は
```
wget  http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
python main.py \
  -m vgg \
  -i 元画像パス \
  -s 画風画像パス
```