from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from os.path import join, relpath
from glob import glob
from PIL import Image
import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('directory', 'data', """Directory where to write *.tfrecords.""")


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(dataset, name):
    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)

    for data in dataset:
        image_path = data[0]
        label = int(data[1])

        image_object = Image.open(image_path)
        image = np.array(image_object)

        height = image.shape[0]
        width = image.shape[1]
        depth = 3
        image_raw = image.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())

    writer.close()


def main(unused_argv):
    if not tf.gfile.Exists(FLAGS.directory):
        tf.gfile.MakeDirs(FLAGS.directory)

    label_data = [
        ['dog', 0],
        ['cat', 1]
    ]

    img_data = []
    for n, v in label_data:
        path = os.path.join('images', n)
        
        for file in [relpath(x, path) for x in glob(join(path, '*.jpg'))]:
            img_data.append([os.path.join(path, file), v])

    convert_to(img_data, 'train')


if __name__ == '__main__':
    tf.app.run()