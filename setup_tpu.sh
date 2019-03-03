#!/bin/bash

hdir=/home/vkuznet
export STORAGE_BUCKET=gs://dataset300
export PATH=$hdir/anaconda2/bin:$PATH
source activate py27
export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$hdir/Particle-Discovery:$PWD/DenseNet
