#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable-msg=
"""
File       : keras_dn.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: 
    the code to train HEP images using DenseNet in TF keras implementation.
    This code is based on TF TPU example:
    https://github.com/tensorflow/tpu/blob/master/tools/colab/keras_mnist_tpu.ipynb
    and, on Keras DenseNet package:
    https://github.com/titu1994/DenseNet
"""

# system modules
import os
import sys
import math
import time
import argparse

# The GPU id to use, usually either "0" or "1";
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# import numpy
import numpy as np

# import matplotlib
import matplotlib
matplotlib.use('Agg') # avoid using XWindows when no DISPLAY is set
import matplotlib.pyplot as plt

# import keras, sklearn
from sklearn import metrics

# import densenet, we use local module tf_densenet which is modification of
# https://github.com/titu1994/DenseNet where keras layers are replaced with
# tf.keras ones.
import tf_densenet as densenet

# import tensorflow
import tensorflow as tf

class OptionParser():
    def __init__(self):
        "User based option parser"
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument("--fdir", action="store",
            dest="fdir", default="", help="Input directory of data images")
        self.parser.add_argument("--fout", action="store",
            dest="fout", default="", help="Output file")
        self.parser.add_argument("--batch_size", action="store",
            dest="batch_size", default=10, help="Batch size")
        self.parser.add_argument("--classes", action="store",
            dest="classes", default=0, help="Number of classification classes")
        self.parser.add_argument("--image-shape", action="store",
            dest="image_shape", default="300,300,3", help="Image shape, default 300,300,3 (color PNG)")
        self.parser.add_argument("--dropout", action="store",
            dest="dropout", default="0.1", help="dropout rate, default 0.1")
        self.parser.add_argument("--epochs", action="store",
            dest="epochs", default=10, help="Number of epochs, default 10")
        self.parser.add_argument("--steps", action="store",
            dest="steps", default=0, help="Number of steps per epoch, default 0")
        self.parser.add_argument("--verbose", action="store_true",
            dest="verbose", default=False, help="verbose output")

def plot_roc_curve(fpr, tpr, fout, title='Keras'):
    auc = metrics.auc(fpr, tpr)
    fout = fout.split('.')[0]
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='{} (area = {:.3f})'.format(title, auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig('{}-roc.pdf'.format(fout))
    plt.close()

def plot_acc(epochs, history, fname):
    acc_values = history['acc']
    plt.figure()
    plt.plot(epochs, history['acc'], label='Training accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('{}-acc.pdf'.format(fname))
    plt.close()

def plot_loss(epochs, history, fname):
    plt.figure()
    plt.plot(epochs, history['loss'], label='Training loss')
    plt.plot(epochs, history['val_loss'], label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('{}-loss.pdf'.format(fname))
    plt.close()

def dataset_to_numpy(dataset, N=10):
    unbatched_ds = dataset.apply(tf.data.experimental.unbatch())
    images, labels = unbatched_ds.batch(N).make_one_shot_iterator().get_next()
    # get one batch 
#    images, labels = dataset.make_one_shot_iterator().get_next()
  
    # Run once, get one batch. Session.run returns numpy results
    with tf.Session() as sess:
        data, classes = sess.run([images, labels])
    return data, classes

def parse_fn(filename, label):
    image_string = tf.read_file(filename, "file_reader")
    image_decoded = tf.image.decode_image(image_string)
    image = tf.cast(image_decoded, tf.float32)
    # https://github.com/tensorflow/tensorflow/issues/16052
    image.set_shape([300,300,3])
    label.set_shape([None, ])
    return image, label

def get_files_labels(fdir):
    samples = [d for d in os.listdir(fdir) if not d.startswith('.')]
    files = []
    labels = []
    for idx, sdir in enumerate(os.listdir(fdir)):
        if sdir.startswith('.'):
            continue
        idir = os.path.join(fdir, sdir)
        if os.path.isdir(idir):
            filenames = [os.path.join(idir, f) for f in os.listdir(idir)]
            files += filenames
            labels += [idx]*len(filenames)
        else:
            files.append(idir)
            labels.append(0)
    return files, labels

def ds_dim(fdir):
    files, _ = get_files_labels(fdir)
    return len(files)

def get_labels(fdir):
    _, labels = get_files_labels(fdir)
    return labels

def get_dataset(fdir, batch_size, img_shape, classes, shuffle=True, cache=False):
    files, labels = get_files_labels(fdir)
    print("input directory: {}".format(fdir))
    print("total files {}, total labels {}".format(len(files), len(labels)))
    print("one file {} and label {}".format(files[0], labels[0]))
    labels = tf.keras.utils.to_categorical(labels, num_classes=classes)
    images = tf.convert_to_tensor(files)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if cache:
        # this small dataset can be entirely cached in RAM, for TPU this is important
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    dataset = dataset.map(parse_fn)
    # drop_remainder is important on TPU, batch size must be fixed
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat() # Mandatory for Keras for now
    dataset = dataset.prefetch(batch_size)  # fetch next batches while training on the current one
    return dataset

def train(fdir, batch_size, image_shape, classes, fout, epochs=10, dropout=0.1, steps_per_epoch=None):

    # input parameters
    train_dir = os.path.join(fdir, 'train')
    valid_dir = os.path.join(fdir, 'valid')
    test_dir = os.path.join(fdir, 'test')
    if not steps_per_epoch:
        steps_per_epoch = 6000//batch_size  # 60,000 items in this dataset
    tpu = None

    # build and train model
    model = densenet.DenseNetImageNet161(input_shape=image_shape,
            classes=classes,  dropout_rate=dropout, weights=None)
#    model = densenet.DenseNet(input_shape=image_shape,
#            classes=classes,  dropout_rate=dropout, weights=None)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    trained_model = model

    # print model layers
    model.summary()

    print("### model", type(model), type(model.fit), model.fit)

    # set up learning rate decay
    lr_decay = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0001 + 0.02 * math.pow(0.5, 1+epoch), verbose=True)

    training_dataset = get_dataset(train_dir, batch_size, image_shape, classes)
    validation_dataset = get_dataset(valid_dir, batch_size, image_shape, classes)
    # test_dataset we'll use in prediction there is no need to shuffle it
    test_dataset = get_dataset(test_dir, batch_size, image_shape, classes, shuffle=False)

    try: # TPU detection
        tpu = tf.contrib.cluster_resolver.TPUClusterResolver() # Picks up a connected TPU on Google's Colab, ML Engine, Kubernetes and Deep Learning VMs accessed through the 'ctpu up' utility
      #tpu = tf.contrib.cluster_resolver.TPUClusterResolver('MY_TPU_NAME') # If auto-detection does not work, you can pass the name of the TPU explicitly (tip: on a VM created with "ctpu up" the TPU has the same name as the VM)
    except ValueError:
        print('Training on GPU/CPU')
        # printout how our session is configured
#        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    print("trainint dataset", type(training_dataset), training_dataset)
    print("validation dataset", type(validation_dataset), validation_dataset)
    print("test dataset", type(test_dataset), test_dataset)

    if tpu: # TPU training
        # For TPU, we will need a function that returns the dataset
        training_input_fn = lambda: get_dataset(train_dir, batch_size, image_shape, classes)
        validation_input_fn = lambda: get_dataset(valid_dir, batch_size, image_shape, classes)

        strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu)
        trained_model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
        # Work in progress: reading directly from dataset object not yet implemented
        # for Keras/TPU. Keras/TPU needs a function that returns a dataset.
        fit = trained_model.fit(training_input_fn,
                steps_per_epoch=steps_per_epoch, epochs=epochs,
                validation_data=validation_input_fn, validation_steps=1, callbacks=[lr_decay])
    else: # GPU/CPU training
        fit = trained_model.fit(training_dataset,
            steps_per_epoch=steps_per_epoch, epochs=epochs,
            validation_data=validation_dataset, validation_steps=1,
            callbacks=[lr_decay])
    
    print("history keys {}".format(fit.history.keys()))
#    print("accuracy: train={} valid={}".format(fit.history['acc'], fit.history['val_acc']))
#    print("loss: train={} valid={}".format(fit.history['loss'], fit.history['val_loss']))

    if fout:
        fname = fout.split(".")[0]
        loss_values = fit.history['loss']
        epochs = range(1, len(loss_values)+1)

        # make plots
        plot_loss(epochs, fit.history, fname)
        plot_acc(epochs, fit.history, fname)

        # choose which dataset/labels we'll use to test our predictions
        # we can use either valid or test datasets
#        tdir = valid_dir
#        tdataset = validation_dataset
        tdir = test_dir
        tdataset = test_dataset
        y_true = get_labels(tdir)
        steps = len(y_true)/batch_size
        probs = model.predict(tdataset, steps=steps)
        y_pred = np.argmax(probs, axis=1)
        print("probs", type(y_true), np.shape(y_true), type(y_pred), np.shape(y_pred))
        print("y_true", y_true[:10])
        print("y_pred", y_pred[:10])
        print("probs ", probs[:10])
        tsize = min(len(y_true), len(y_pred)) - 1
        y_true = y_true[:tsize]
        y_pred = y_pred[:tsize]
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        print(metrics.confusion_matrix(y_true, y_pred))
        plot_roc_curve(fpr, tpr, fname, title='Keras')

def main():
    "Main function"
    optmgr  = OptionParser()
    opts = optmgr.parser.parse_args()
    fdir = opts.fdir
    if not fdir:
        print("please setup fdir")
        sys.exit(1)
    batch_size = int(opts.batch_size)
    image_shape = tuple([int(s) for s in opts.image_shape.split(',')])
    classes = int(opts.classes)
    epochs = int(opts.epochs)
    dropout = float(opts.dropout)
    fout = opts.fout
    steps = int(opts.steps)
    print("Input parameters")
    print("fdir {}".format(fdir))
    print("batch_size {}".format(batch_size))
    print("image_shape {}".format(image_shape))
    print("classes {}".format(classes))
    print("epochs {}".format(epochs))
    print("dropout {}".format(dropout))
    print("fout {}".format(fout))
    print("steps {}".format(steps))
    if not classes:
        print("please setup number of trained classes")
        sys.exit(1)
    time0 = time.time()
    train(fdir, batch_size, image_shape, classes, fout, epochs, dropout, steps)
    print("Elapsed time: {} sec".format(time.time()-time0))

if __name__ == '__main__':
    main()
