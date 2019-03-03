#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable-msg=
"""
File       : img2tfrecs.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: 
    based on
    https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
    https://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
"""

# system modules
import os
import sys
import argparse

# image module
from PIL import Image

# numpy module
import numpy as np

# tensorflow module
import tensorflow as tf

class OptionParser():
    def __init__(self):
        "User based option parser"
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument("--fdir", action="store",
            dest="fdir", default="", help="Input directory of data images")
        self.parser.add_argument("--image-shape", action="store",
            dest="image_shape", default="300,300,3", help="Image shape, default 300,300,3 (color PNG)")
        self.parser.add_argument("--fout", action="store",
            dest="fout", default="", help="Output file")
        self.parser.add_argument("--verbose", action="store_true",
            dest="verbose", default=False, help="verbose output")

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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

def convert_to(fdir, fout, img_shape, verbose=False):
    """Converts a dataset to tfrecords."""
    files, labels = get_files_labels(fdir)
    if not fout.endswith('.tfrecords'):
        fout = fout+'.tfrecords'
    rows = img_shape[0]
    cols = img_shape[1]
    depth = img_shape[2]
    if verbose:
        print("output: {}".format(fout))
    with tf.python_io.TFRecordWriter(fout) as writer:
        for idx, fname in enumerate(files):
            img = np.array(Image.open(fname))
            image = img.tostring()
            if verbose:
                print("fname: {}".format(fname))
            label = labels[idx]
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(rows),
                        'width': _int64_feature(cols),
                        'depth': _int64_feature(depth),
                        'label': _int64_feature(label),
                        'image': _bytes_feature(image)
                    }))
            writer.write(example.SerializeToString())

def main():
    "Main function"
    optmgr  = OptionParser()
    opts = optmgr.parser.parse_args()
    fdir = opts.fdir
    fout = opts.fout
    img_shape = tuple([int(s) for s in opts.image_shape.split(',')])
    verbose = opts.verbose
    convert_to(fdir, fout, img_shape, verbose)

if __name__ == '__main__':
    main()
