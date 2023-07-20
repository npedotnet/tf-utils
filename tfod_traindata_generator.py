#
# tfod_traindata_generator.py
#
# Train data generator for TensorFlow Object Detection API.
# MIT LICENSE
#
# Usage: python tfod_traindata_generator.py [input_dir] [train_rate]
# For more details, see https://github.com/npedotnet/tf-utils
#
# 日本語の詳細はこちらをご覧下さい
# https://unixo-lab/unity/vott.html#学習に必要なデータを準備する
#

import tensorflow as tf

IMAGE_ENCODED = 'image/encoded'
IMAGE_FILENAME = 'image/filename'
IMAGE_FORMAT = 'image/format'
IMAGE_HEIGHT = 'image/height'
IMAGE_KEY_SHA256 = 'image/key/sha256'
IMAGE_OBJECT_BBOX_XMAX = 'image/object/bbox/xmax'
IMAGE_OBJECT_BBOX_XMIN = 'image/object/bbox/xmin'
IMAGE_OBJECT_BBOX_YMAX = 'image/object/bbox/ymax'
IMAGE_OBJECT_BBOX_YMIN = 'image/object/bbox/ymin'
IMAGE_OBJECT_CLASS_LABEL = 'image/object/class/label'
IMAGE_OBJECT_CLASS_TEXT = 'image/object/class/text'
IMAGE_OBJECT_DIFFICULT = 'image/object/difficult'
IMAGE_OBJECT_TRUNCATED = 'image/object/truncated'
IMAGE_OBJECT_VIEW = 'image/object/view'
IMAGE_SOURCEID = 'image/source_id'
IMAGE_WIDTH = 'image/width'

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def load(files):
    features = {
        IMAGE_ENCODED: tf.io.FixedLenFeature([], tf.string),
        IMAGE_FILENAME: tf.io.FixedLenFeature([], tf.string),
        IMAGE_FORMAT: tf.io.FixedLenFeature([], tf.string),
        IMAGE_HEIGHT: tf.io.FixedLenFeature([], tf.int64),
        IMAGE_KEY_SHA256: tf.io.FixedLenFeature([], tf.string),
        IMAGE_OBJECT_BBOX_XMAX: tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        IMAGE_OBJECT_BBOX_XMIN: tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        IMAGE_OBJECT_BBOX_YMAX: tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        IMAGE_OBJECT_BBOX_YMIN: tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        IMAGE_OBJECT_CLASS_LABEL: tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        IMAGE_OBJECT_CLASS_TEXT: tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        IMAGE_OBJECT_DIFFICULT: tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        IMAGE_OBJECT_TRUNCATED: tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        IMAGE_OBJECT_VIEW: tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        IMAGE_SOURCEID: tf.io.FixedLenFeature([], tf.string),
        IMAGE_WIDTH: tf.io.FixedLenFeature([], tf.int64),
    }
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, features))
    return dataset

def export(data):
    encoded = data[IMAGE_ENCODED].numpy()
    filename = data[IMAGE_FILENAME].numpy()
    format = data[IMAGE_FORMAT].numpy()
    height = data[IMAGE_HEIGHT].numpy()
    sha256 = data[IMAGE_KEY_SHA256].numpy()
    xmax = data[IMAGE_OBJECT_BBOX_XMAX].numpy()
    xmin = data[IMAGE_OBJECT_BBOX_XMIN].numpy()
    ymax = data[IMAGE_OBJECT_BBOX_YMAX].numpy()
    ymin = data[IMAGE_OBJECT_BBOX_YMIN].numpy()
    label = data[IMAGE_OBJECT_CLASS_LABEL].numpy()
    text = data[IMAGE_OBJECT_CLASS_TEXT].numpy()
    difficult = data[IMAGE_OBJECT_DIFFICULT].numpy()
    truncated = data[IMAGE_OBJECT_TRUNCATED].numpy()
    view = data[IMAGE_OBJECT_VIEW].numpy()
    source_id = data[IMAGE_SOURCEID].numpy()
    width = data[IMAGE_WIDTH].numpy()
    feature = {
        IMAGE_ENCODED: bytes_feature(encoded),
        IMAGE_FILENAME: bytes_feature(filename),
        IMAGE_FORMAT: bytes_feature(format),
        IMAGE_HEIGHT: int64_feature(height),
        IMAGE_KEY_SHA256: bytes_feature(sha256),
        IMAGE_OBJECT_BBOX_XMAX: float_list_feature(xmax),
        IMAGE_OBJECT_BBOX_XMIN: float_list_feature(xmin),
        IMAGE_OBJECT_BBOX_YMAX: float_list_feature(ymax),
        IMAGE_OBJECT_BBOX_YMIN: float_list_feature(ymin),
        IMAGE_OBJECT_CLASS_LABEL: int64_list_feature(label),
        IMAGE_OBJECT_CLASS_TEXT: bytes_list_feature(text),
        IMAGE_OBJECT_DIFFICULT: int64_list_feature(difficult),
        IMAGE_OBJECT_TRUNCATED: int64_list_feature(truncated),
        IMAGE_OBJECT_VIEW : bytes_list_feature(view),
        IMAGE_SOURCEID: bytes_feature(filename),
        IMAGE_WIDTH: int64_feature(width),
    }
    features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=features)
    return example.SerializeToString()

def save(dataset, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for data in dataset:
            writer.write(export(data))

# Entry point

import sys
import glob

args = sys.argv

if(len(args) != 3):
    print("Usage: tfod_traindata_generator.py [input_dir] [train_rate]")
    exit()

files = glob.glob(args[1] + '/*.tfrecord')

separator = (int)((float)(args[2]) * len(files))

# train
train_dataset = load(files[:separator])
save(train_dataset, 'train.tfrecord')

# val
val_dataset = load(files[separator:])
save(val_dataset, 'val.tfrecord')
