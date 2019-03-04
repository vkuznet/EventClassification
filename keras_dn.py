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

Additional resouces for TPU and TF datasets:
    https://www.tensorflow.org/guide/using_tpu
    https://www.tensorflow.org/guide/performance/datasets
    https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    https://github.com/tensorflow/models/blob/master/official/mnist/mnist_tpu.py
    https://www.tensorflow.org/guide/datasets
    https://www.tensorflow.org/guide/performance/datasets
    https://www.tensorflow.org/tutorials/load_data/tf_records
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
np.random.seed(123)

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

# global variables
IMG_SHAPE = None
NCLASSES = None

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
        self.parser.add_argument("--test", action="store_true",
            dest="test", default=False, help="use test DenseNet model")
        self.parser.add_argument("--verbose", action="store_true",
            dest="verbose", default=False, help="verbose output")

def plot_roc_curve(fpr, tpr, fout, title='Keras'):
    "Helper function to plot roc curve"
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
    "Helper function to plot trainining accuracies"
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
    "Helper function to plot training losses"
    plt.figure()
    plt.plot(epochs, history['loss'], label='Training loss')
    plt.plot(epochs, history['val_loss'], label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('{}-loss.pdf'.format(fname))
    plt.close()

def dataset_to_numpy(dataset, nentries=10):
    """
    Helper function to return numpy representation of a TF dataset (images, labels)
    """
    unbatched_ds = dataset.apply(tf.data.experimental.unbatch())
    images, labels = unbatched_ds.batch(nentries).make_one_shot_iterator().get_next()
    # get one batch 
#    images, labels = dataset.make_one_shot_iterator().get_next()
  
    # Run once, get one batch. Session.run returns numpy results
    with tf.Session() as sess:
        data, classes = sess.run([images, labels])
    return data, classes

def parse_fn(filename, label):
    "Helper function to return image and label from given file/label pair"
    image_string = tf.read_file(filename, "file_reader")
    image_decoded = tf.image.decode_image(image_string)
    image = tf.cast(image_decoded, tf.float32)
    # we still need to explicitly set shape of the image
    # https://github.com/tensorflow/tensorflow/issues/16052
    global IMG_SHAPE
    image.set_shape(list(IMG_SHAPE))
    label.set_shape([None, ])
    return image, label

def get_files_labels(fdir):
    """
    Helper function to return files and labels from given input directory.
    It can read either fdir/{train,test,valid}/{Sample1,Sample2}
    structure or fdir/{train,test,valid}/*.tfrecords

    It also takes care to replace fdir with proper STORAGE_BUCKET
    environment value (if it is set) which is useful when working on
    Google Cloud platform which requires to read data from Google Storage.
    """
    samples = [d for d in os.listdir(fdir) if not d.startswith('.')]
    files = []
    labels = []
    for idx, sdir in enumerate(os.listdir(fdir)):
        if sdir.startswith('.'):
            continue
        idir = os.path.join(fdir, sdir)
        if os.path.isdir(idir):
            # if TPU or google cloud is present
            if 'STORAGE_BUCKET' in os.environ:
                tdir = idir.replace(fdir, os.environ['STORAGE_BUCKET'])
                for sd in ['train', 'valid', 'test']:
                    if sd in fdir:
                        tdir = os.path.join(os.environ['STORAGE_BUCKET'], sd)
                tdir = os.path.join(tdir, sdir)
                filenames = [os.path.join(tdir, f) for f in os.listdir(idir)]
            else:
                filenames = [os.path.join(idir, f) for f in os.listdir(idir)]
            files += filenames
            labels += [idx]*len(filenames)
        else:
            # labels here is irrelevant since we assume files
            # contains all the information, e.g. tfrecords
            files.append(idir)
            labels.append(0)
    return files, labels

def ds_dim(fdir):
    "Helper function to return dimension of dataset (total number of files)"
    files, _ = get_files_labels(fdir)
    return len(files)

def get_labels(fdir):
    "Helper function to get labels from given input directory of images"
    _, labels = get_files_labels(fdir)
    return labels

def get_dataset(fdir, batch_size, shuffle=False, cache=False, tpu=False):
    """
    Top level function to create TF dataset from given input directory.
    This directory can either contain images or tfrecords files
    """
    files, labels = get_files_labels(fdir)
    print("input directory: {}".format(fdir))
    print("total files {}, total labels {}".format(len(files), len(labels)))
    print("one file {} and label {}".format(files[0], labels[0]))
    if files[0].endswith('tfrecords'):
        return get_tfrecords(files, batch_size, shuffle, cache, tpu)
    return get_dataset_img(files, labels, batch_size, shuffle, cache, tpu)

def get_dataset_img(files, labels, batch_size, shuffle=False, cache=False, tpu=False):
    "Helper function to create TF dataset from given set of files and labels"
    global NCLASSES
    labels = tf.keras.utils.to_categorical(labels, num_classes=NCLASSES)
    images = tf.convert_to_tensor(files)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if cache:
        # this small dataset can be entirely cached in RAM, for TPU this is important
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    if tpu:
        # for TPU it is important to delivery data fast, we need to choose
        # number of parallel calls appropriately, see
        # https://www.tensorflow.org/guide/performance/datasets
        dataset = dataset.map(parse_fn, num_parallel_calls=10)
    else:
        dataset = dataset.map(parse_fn)
    # drop_remainder is important on TPU, batch size must be fixed
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat() # Mandatory for Keras for now
    dataset = dataset.prefetch(batch_size)  # fetch next batches while training on the current one
    return dataset

def parse_tfrec(example_proto):
    """
    Helper function to parse tfrecords (see img2tfrecs.py which creates them
    from images). We assume that tfrecords provides the image parameters
    and label.
    """
    features={
        'height': tf.FixedLenFeature([], tf.int64, default_value=IMG_SHAPE[0]),
        'width': tf.FixedLenFeature([], tf.int64, default_value=IMG_SHAPE[1]),
        'depth': tf.FixedLenFeature([], tf.int64, default_value=IMG_SHAPE[2]),
        'label': tf.FixedLenFeature([], tf.int64, default_value=0),
        'image': tf.FixedLenFeature([], tf.string, default_value=''),
        }
    parsed_features = tf.parse_single_example(example_proto, features)
    height = tf.cast(parsed_features['height'], tf.int32)
    width = tf.cast(parsed_features['width'], tf.int32)
    depth = tf.cast(parsed_features['depth'], tf.int32)
    label = tf.cast(parsed_features['label'], tf.int32)
    # convert label numerical value into categorical vector based on NCLASSES
    global NCLASSES
    label = tf.one_hot(label, NCLASSES)
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    image_shape = tf.stack([height, width, depth])
    image = tf.reshape(image, image_shape)
    # we still need to explicitly set shape of the image
    # https://github.com/tensorflow/tensorflow/issues/16052
    image.set_shape(list(IMG_SHAPE))
    return image, label

def get_tfrecords(files, batch_size, shuffle=False, cache=False, tpu=False):
    """
    Helper function which allows to read tfrecords input files and return
    TF dataset which contains images and labels. The labels are converted
    from numerical values to categorical vector.
    """
    dataset = tf.data.TFRecordDataset(files)
    if cache:
        # this small dataset can be entirely cached in RAM, for TPU this is important
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    if tpu:
        # for TPU it is important to delivery data fast, we need to choose
        # number of parallel calls appropriately, see
        # https://www.tensorflow.org/guide/performance/datasets
        dataset = dataset.map(parse_tfrec, num_parallel_calls=10)
    else:
        dataset = dataset.map(parse_tfrec)
    # drop_remainder is important on TPU, batch size must be fixed
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat() # Mandatory for Keras for now
    dataset = dataset.prefetch(batch_size)  # fetch next batches while training on the current one
    return dataset

def train(fdir, batch_size, image_shape, classes, fout, epochs=10, dropout=0.1,
        steps_per_epoch=None, is_test=False):
    """
    Main function which does the training of our ML model either from
    images or tfrecords from provided input directory fdir.
    """

    # input parameters
    train_dir = os.path.join(fdir, 'train')
    valid_dir = os.path.join(fdir, 'valid')
    test_dir = os.path.join(fdir, 'test')
    if not steps_per_epoch:
        steps_per_epoch = 6000//batch_size  # 60,000 items in this dataset
    tpu = None

    # build and train model
    if is_test:
        # for testing use very small DenseNet which is defined by depth parameter
        model = densenet.DenseNet(input_shape=image_shape, depth=10,
                classes=classes,  dropout_rate=dropout, weights=None)
    else:
        model = densenet.DenseNetImageNet161(input_shape=image_shape,
                classes=classes,  dropout_rate=dropout, weights=None)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    trained_model = model

    # print model layers
    model.summary()

    print("### model", type(model), type(model.fit), model.fit)

    # set up learning rate decay
    lr_decay = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0001 + 0.02 * math.pow(0.5, 1+epoch), verbose=True)

    training_dataset = get_dataset(train_dir, batch_size, image_shape, classes)
    validation_dataset = get_dataset(valid_dir, batch_size, image_shape, classes)
    test_dataset = get_dataset(test_dir, batch_size, image_shape, classes)

    try: # TPU detection
        # Picks up a connected TPU on Google's Colab, ML Engine, Kubernetes
        # and Deep Learning VMs accessed through the 'ctpu up' utility
        tpu = tf.contrib.cluster_resolver.TPUClusterResolver()
        # If auto-detection does not work, you can pass the name of the TPU explicitly
        # on a VM created with "ctpu up" the TPU has the same name as the VM
#        tpu = tf.contrib.cluster_resolver.TPUClusterResolver('MY_TPU_NAME')
        print('### Training on TPU ###')
    except ValueError:
        print('### Training on GPU/CPU ###')
        # printout how our session is configured
#        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    print("trainint dataset", type(training_dataset), training_dataset)
    print("validation dataset", type(validation_dataset), validation_dataset)
    print("test dataset", type(test_dataset), test_dataset)

    if tpu: # TPU training
        # For TPU, we will need a function that returns the dataset
        training_input_fn = lambda: get_dataset(train_dir, batch_size,
                image_shape, classes, tpu=True)
        validation_input_fn = lambda: get_dataset(valid_dir, batch_size,
                image_shape, classes, tpu=True)

        strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu)
        trained_model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
        # Work in progress: reading directly from dataset object not yet implemented
        # for Keras/TPU. Keras/TPU needs a function that returns a dataset.
        fit = trained_model.fit(training_input_fn,
                steps_per_epoch=steps_per_epoch, epochs=epochs,
                validation_data=validation_input_fn, validation_steps=1,
                verbose=1,
                callbacks=[lr_decay])
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
        tdir = test_dir
        tdataset = test_dataset
        y_true = get_labels(tdir)
        steps = len(y_true)/batch_size
        if not steps:
            steps = 1
        if tpu:
            # so far predictions does not work on TPU I need to figure
            # out what to pass to predict method, and goal here is to train
            # the model which later can be used for inference
            return
            input_fn = lambda: get_dataset(tdir, batch_size, image_shape, classes, tpu=True)
            probs = trained_model.predict(input_fn, steps=steps)
        else:
            probs = trained_model.predict(tdataset, steps=steps)
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
    global IMG_SHAPE
    IMG_SHAPE = image_shape
    classes = int(opts.classes)
    global NCLASSES
    NCLASSES = classes
    epochs = int(opts.epochs)
    dropout = float(opts.dropout)
    fout = opts.fout
    steps = int(opts.steps)
    is_test = opts.test
    print("{}\n".format(' '.join(sys.argv)))
    print("Input parameters")
    print("fdir        {}".format(fdir))
    print("batch_size  {}".format(batch_size))
    print("image_shape {}".format(image_shape))
    print("classes     {}".format(classes))
    print("epochs      {}".format(epochs))
    print("dropout     {}".format(dropout))
    print("fout        {}".format(fout))
    print("steps       {}".format(steps))
    if not classes:
        print("please setup number of trained classes")
        sys.exit(1)
    time0 = time.time()
    train(fdir, batch_size, image_shape, classes, fout, epochs, dropout, steps, is_test)
    print("Elapsed time: {} sec".format(time.time()-time0))

if __name__ == '__main__':
    main()
