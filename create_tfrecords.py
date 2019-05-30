# coding=utf-8
import os
import tensorflow as tf
from PIL import Image
import sys


def creat_tf(imgpath):
    classes = os.listdir(imgpath)

    writer = tf.python_io.TFRecordWriter("train.tfrecords")

    for index, name in enumerate(classes):
        class_path = imgpath + name
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = class_path + '/' + img_name
                img = Image.open(img_path)
                # you can improve, not resize
                img = img.resize((224, 224))
                img_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())
    writer.close()


def read_example():
    for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        # image = example.features.feature['img_raw'].bytes_list.value
        label = example.features.feature['label'].int64_list.value
        print(label[0])


if __name__ == '__main__':
    imgpath = './Data/'
    creat_tf(imgpath)
    # read_example()
