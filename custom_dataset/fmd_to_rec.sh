#!/bin/bash

wget https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip
unzip FMD.zip
m="/Users/slakshay/anaconda3/lib/python3.6/site-packages/mxnet"
python $m/tools/im2rec.py --list --recursive --train-ratio 0.7 fmd image
python $m/tools/im2rec.py --resize 240 --quality 95 --num-thread 16 fmd image