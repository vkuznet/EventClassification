#!/bin/bash
export PATH=/mnt/data1/vk/anaconda2/bin:$PATH
source activate myenv
nvidia-smi -L

echo "Example how to run densenet"
echo "time python train.py --PATH_data /mnt/data1/vk/dl_exploration/data/all_vs_all --PATH_save_images /mnt/data1/vk/RandD/report/ --means 0.002886 0.015588 0.016239 --stdevs 0.052924 0.123025 0.125617 --fc_features 19872 --gpus 0 2>&1 1>& job.log < /dev/null &"

# example from goofy for 600x600 imags
# nohup python train.py --gpus 2 --bs 8 --epochs_last_layer 2 --epochs 10
# --shape 600 --betas .9 .999 --lr 1e-4 --weight_decay 0 --dropout .5
# --PATH_data=/share/lazy/vk/data/images_600/higgs_vs_all
# --PATH_save_images=/share/lazy/vk/data/dl_exploration --pretrained
# --fc_features 317952 --means 0.002886 0.015588 0.016239 --stdevs 0.052924
# 0.123025 0.125617 2>&1 1>& log < /dev/null &
