#!/bin/sh

python main.py \
  -m nin \
  -i ~/git/chainer-gogh/sample_images/cat.png \
  -s ~/git/chainer-gogh/sample_images/style_0.png \
  -o tmp \
  --iter 100 \
  --lr 4.0 \
  --lam 0.01 \
  --width 100 \
  --height 100 \
  --no_resize_style

