import os

import models
from data import load_image
from config import Config, parse_args


def main():
  args = parse_args()
  config = Config(args)

  # 出力先の作成
  os.makedirs(config.output_dir, exist_ok=True)

  # モデルの作成
  model = models.generate_model(config.model)

  # 画像サイズの修正
  img_orig = load_image(config.original_image, [config.width, config.height])
  img_style = load_image(config.style_image, [config.width, config.height] if not config.no_resize_style else None)

  # 画像を生成
  generator = models.Generator(model, img_orig, img_style, config)
  generator.generate(config)


if __name__ == '__main__':
  main()
