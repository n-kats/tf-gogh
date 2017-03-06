import models
from data import load_image
from config import Config
import argparse
import os


def parse_args():
  parser = argparse.ArgumentParser(description='chainer-goghの真似')
  parser.add_argument('--model', '-m', default='nin',
                      help='nin, vggのどちらか')
  parser.add_argument('--orig_img', '-i', help="入力の画像")
  parser.add_argument('--style_img', '-s', help="画風の画像")
  parser.add_argument('--out_dir', '-o', default="tmp/output", help="生成画像の保存先")
  parser.add_argument('--iter', default=5000, type=int, help="学習回数")
  parser.add_argument('--lr', default=4.0, type=float, help="学習レート")
  parser.add_argument('--lam', default=0.005, type=float, help="入力と画風のバランス")
  parser.add_argument('--width', '-w', default=435, type=int, help="生成画像のサイズ")

  args = parser.parse_args()
  return args


def main():
  args = parse_args()
  config = Config(args)

  # 出力先の作成
  os.makedirs(args.out_dir, exist_ok=True)

  # モデルの作成
  model = models.generate_model(args.model)

  # 画像サイズの修正
  img_orig = load_image(args.orig_img, [args.width, args.width])
  img_style = load_image(args.style_img, [args.width, args.width])

  # 画像を生成
  generator = models.Generator(model, img_orig, img_style, config)
  generator.generate(config)

if __name__ == '__main__':
  main()
