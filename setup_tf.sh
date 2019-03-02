#!/bin/bash

export PATH=/mnt/data1/vk/anaconda2/bin:$PATH
source activate py27
nvidia-smi -L
export PYTHONPATH=$PWD:$PYTHONPATH:/mnt/data1/vk/RandD/DenseNet
