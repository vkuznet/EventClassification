#!/bin/bash

# QuickStart: https://cloud.google.com/tpu/docs/quickstart

# MNIST tutorial on TPU
# https://cloud.google.com/tpu/docs/tutorials/mnist

# setup proper zones to use TPU
gcloud config set compute/zone us-central1-f
ctpu up -zone us-central1-f

# download our data
scp vek3@lnxcu9.lns.cornell.edu:/mnt/disk1/vk/dataset300_test.tar.gz .
tar xfz dataset300_test.tar.gz
rm dataset300_test.tar.gz

# copy our dataset to google storage
# first I need to create storage bucket at
# https://console.cloud.google.com/storage/browser
export STORAGE_BUCKET=gs://dataset300
gsutil cp -r dataset300_test/* ${STORAGE_BUCKET}
#gsutil -m cp -r dataset300_test/* ${STORAGE_BUCKET}

# download and install conda
curl -L -O https://repo.anaconda.com/archive/Anaconda2-2018.12-Linux-x86_64.sh
sh Anaconda2-2018.12-Linux-x86_64.sh

export PATH=/home/vkuznet/anaconda2/bin:$PATH

# setup envronment
conda update conda
conda create --name py27 python=2.7

# setup environment to point to anaconda
source activate py27

# install required packages, I'm not sure if I need to isntall tensorflow-gpu
# since it is provided on google cloud nodes
#conda install keras matplotlib scikit-learn tensorflow-gpu
conda install keras matplotlib scikit-learn

# download DenseNet
git clone https://github.com/titu1994/DenseNet.git
git clone https://github.com/vkuznet/Particle-Discovery.git

# setup PYTHONPATH
export PYTHONPATH=$PWD/Particle-Discovery:$PYTHONPATH:$PWD/DenseNet

cd Particle-Discovery
# reset PYTHONPATH to TPU tf
export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHONPATH

# test run
echo "./keras_dn.py --fdir=${STORAGE_BUCKET}/dataset300_test --classes=2 --epochs=1 --dropout=0.5 --fout=keras --batch_size=20 --steps=100"

# full run
#nohup ./keras_dn.py --fdir=${STORAGE_BUCKET}/dataset300_test --classes=2 --epochs=10 --dropout=0.5 --fout=keras --batch_size=20 --steps=19000 2>&1 1>& log < /dev/null &

### IMPORTANT:
# by the end of work disable TPUs

# exit from VM
exit

# release our TPU
ctpu delete -zone us-central1-f

# release our VM
ctpu status -zone us-central1-f

# delete storage bucket
gsutil -m rm -r gs://dataset300/{train,valid,test}
