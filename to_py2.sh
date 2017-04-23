mkdir -p _py2
rm _py2/*

cp *.py _py2
cd _py2

3to2 -w -n *.py

for file in `\find . -name '*.py'`; do
  sed -i '1s/^/# -*- coding: utf-8 -*-\n/' $file
  sed -i -e 's/print \(.*\)$/print(\1)/g' $file
  sed -i -e 's/from itertools import izip//g' $file
  sed -i -e 's/izip/zip/g' $file
  sed -i -e 's/xrange/range/g' $file
done
cd ..
