from resnet_train import train
import tensorflow as tf
import time
import os
import sys
import re
import numpy as np

from synset import *
from image_processing import image_preprocessing
from resnet import inference

tf.app.flags.DEFINE_string('train_file','','train file path')
FLAGS = tf.app.flags.FLAGS

def load_data(train_file):
    data = []
    i = 0
    files = []
    labels = []
    start_time = time.time()
    with open(train_file,"r") as fr:
        for line in fr.readlines():
            infos = line.split("\t")   
            data.append({
                "filename": infos[0],
                "label_name": int(infos[1]),
            })

    return data

def load_data_tmp(data_dir):
    data = []
    i = 0
    files = []
    labels = []
    print "listing files in", data_dir
    start_time = time.time()

    for rootpath, dirnames, filenames in os.walk(data_dir):
        for dir0 in dirnames:
            path = os.path.join(data_dir, dir0)
            files0 = os.listdir(path)
            files += [os.path.join(path, item) for item in files0]
            labels += [int(dir0)]*len(files0)

    duration = time.time() - start_time
    print "took %f sec" % duration

    for i in range(len(files)):
        data.append({
            "filename": files[i],
            "label_name": labels[i],
        })

    return data


def distorted_inputs():
    data = load_data(FLAGS.train_file)

    filenames = [ d['filename'] for d in data ]
    label_indexes = [ d['label_name'] for d in data ]

    filename, label_index = tf.train.slice_input_producer([filenames, label_indexes], shuffle=True)

    num_preprocess_threads = 4
    images_and_labels = []
    for thread_id in range(num_preprocess_threads):
        print "filename:",filename
        image_buffer = tf.read_file(filename)

        bbox = []
        train = True
        image = image_preprocessing(image_buffer, bbox, train, thread_id)
        images_and_labels.append([image, label_index])

    images, label_index_batch = tf.train.batch_join(
        images_and_labels,
        batch_size=FLAGS.batch_size,
        capacity=2 * num_preprocess_threads * FLAGS.batch_size)

    height = FLAGS.input_size
    width = FLAGS.input_size
    depth = 3

    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[FLAGS.batch_size, height, width, depth])

    return images, tf.reshape(label_index_batch, [FLAGS.batch_size])

def inputs():
    pass

def main(_):
    images, labels = distorted_inputs()
    is_training = tf.placeholder('bool',[], name='is_training')
    logits = inference(images,
                       num_classes=2,
                       is_training=is_training,
                       bottleneck=True,
                       num_blocks=[3, 4, 6, 3])
    train(is_training,logits, images, labels)


if __name__ == '__main__':
    tf.app.run()
