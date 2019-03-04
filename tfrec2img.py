#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable-msg=
"""
File       : tfrec2img.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: 
"""

# system modules
import os
import sys
import argparse

# third party modules
from PIL import Image
import tensorflow as tf
import numpy as np

class OptionParser():
    def __init__(self):
        "User based option parser"
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument("--fname", action="store",
            dest="fname", default="", help="Input file name of tfrecords file")
        self.parser.add_argument("--fout", action="store",
            dest="fout", default="", help="Output file name for decoded image")

def run(fname, fout):
    """
    Helper function which can decode given tfrecrods file and produce single
    image and information about it. It is doing reverse operation from
    img2tfrecs.py code
    """
    record_iterator = tf.python_io.tf_record_iterator(path=fname)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        depth = int(example.features.feature['depth'].int64_list.value[0])
        label = int(example.features.feature['label'].int64_list.value[0])
        img_string = (example.features.feature['image'].bytes_list.value[0])
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        rec_img = img_1d.reshape((height, width, depth))
        print('shape: ({},{},{}) label: {}'.format(height, width, depth, label))
        if fout:
            img = Image.fromarray(rec_img)
            img.save(fout)
            print('saved image: {}'.format(fout))
        else:
            print('img: {}'.format(rec_img))
        break

def main():
    "Main function"
    optmgr  = OptionParser()
    opts = optmgr.parser.parse_args()
    run(opts.fname, opts.fout)

if __name__ == '__main__':
    main()
