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
tf.app.flags.DEFINE_string('directory', 'data', """Directory where to read *.tfrecords.""")


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })

    image_raw = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)
    label = tf.cast(features['label'], tf.int32)

    image = tf.reshape(image_raw, tf.stack([height, width, depth]))

    return image, label


def inputs():
    if not FLAGS.directory:
        raise ValueError('Please supply a directory')

    tfrecords_filename = 'train'
    filename = os.path.join(FLAGS.directory, tfrecords_filename + '.tfrecords')
    filename_queue = tf.train.string_input_producer([filename])

    image, label = read_and_decode(filename_queue)

    return image, label


def main(unused_argv):
    if not os.path.exists('output'):
        os.mkdir('output')

    images, labels = inputs()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            for i in range(6):
                e, l = sess.run([images, labels])
                img = Image.fromarray(e, 'RGB')
                img.save(os.path.join('output', "{0}-{1}.jpg".format(str(i), l)))
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run()