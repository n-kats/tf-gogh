# 機械学習勉強会資料
## 概要
学習済みモデルを使って画像の特徴を獲得し、二つの画像の特徴を混ぜた画像を作成する。
片方の画像から画風を獲得し、それをもう片方の画像に混ぜる。
![sample.png](https://raw.githubusercontent.com/n-kats/tf-gogh/master/images/sample.png)

[chainer-gogh](https://github.com/pfnet-research/chainer-gogh)を参考にtensorflowで書いたもの。caffemodelを読み込む部分でchainerのcaffemodelを読み込むものを使っているが、それ以外はtensorflow。

## 使い方
systemのpythonを使う、pyenv-virtualenvなどのversion管理ツールのpythonを使う、dockerを使うという方法を想定します。
python3を想定して作ってありますが、_py2ディレクトリにpython2用のものがあります。note.ipynbは、解説用にjupyterでみるためのもので、python3でのみ動作します。

### 共通
* このソースコードをダウンロード。
```
git clone https://github.com/n-kats/tf-gogh.git
cd tf-gogh
```
* 特徴量のパラメータを次のリンクからcaffemodelをダウンロードし、このファイルと同じディレクトリに配置する。(ファイル名を変えないでください)
  * [nin_imagenet.caffemodel](https://gist.github.com/mavenlin/d802a5849de39225bcc6)
  * [VGG_ILSVRC_16_layers.caffemodel](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)
* 画風を変換したい元画像と、画風画像を用意。

### systemなどのpython3を使う場合
* 依存するパッケージをインストールする。いずれもpipでインストールできます。
  * chainer
  * pillow
  * tensorflow
  * jupyter
  * matplotlib

#### CLIを使う場合
main.pyを実行します。画像のパス、画像のサイズ、学習のハイパーパラメータなどを指定します。

例:
```
python3 main.py -i ./images/cat.png \
                -s ./images/gogh.png \
                -m nin \
                --iteration 5000 \
                --lr 4.0 \
                --lam 0.005 \
                --height 300 \
                --width 300
```

`python3 main.py --help`で詳細の説明が読めます。

#### jupyterを使う場合
`jupyter notebook`でjupyterを起動し、note.ipynbを開き、上から順に実行。
トークンを入力せよという画面が出る場合は、コンソールに表示される`http://localhost:8888?token=ほにゃらら`と書かれたリンクを開く。

### python2を使う場合
python2を使う場合は`_py2`ディレクトリのものを使います。
caffemodelファイルを`_py2`ディレクトリに入れてください。
依存するパッケージをインストールしてください
* chainer
* pillow
* tensorflow

あとは、python3のCLIと同様です。

例:
```
python main.py -i ../images/cat.png \
               -s ../images/gogh.png \
               -m nin \
               --iteration 5000 \
               --lr 4.0 \
               --lam 0.005 \
               --height 300 \
               --width 300
```
（画像の相対パスが変わっていることに注意）
こちらでも`python main.py --help`を参考にしてください。

### dockerを使う場合
tensorflow公式のDockerイメージは日本語が使えないので、DockerHubにdockerイメージを用意したのでそれを使ってください。
Dockerイメージはサイズが大きいので会場でダウンロードしないでください。

`docker pull nkats/mln:20170513`
でダウンロードできます。対応するDockerfileは`docker`ディレクトリにあります。
このdockerイメージには、日本語環境、python3とtensorflowとその他もろもろのパッケージが含まれます。

#### CLIで使う場合
```
docker run -v $(pwd):/workspace --rm -it nkats/mln:20170513 \
    python3 main.py -i ./images/cat.png \
                   -s ./images/gogh.png \
                   -m nin \
                   --iteration 5000 \
                   --lr 4.0 \
                   --lam 0.005 \
                   --height 300 \
                   --width 300
```
のようにpython3の場合と同様に実行します。

ヘルプも次のようにして見ることができます。
```
docker run -v $(pwd):/workspace --rm -t nkats/mln:20170513 python3 main.py --help
```

#### jupyterで使う場合
次のコマンドで実行できます。
```
docker run -v $(pwd):/workspace -p 8888:8888 --rm -it nkats/mln:20170513 jupyter notebook --ip=0.0.0.0 --allow-root
```