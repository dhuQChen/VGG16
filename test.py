import tensorflow as tf
from VGG16 import *
import numpy as np
import cv2
import os


def test(path):
    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
    keep_prob = tf.placeholder(tf.float32)
    output = VGG16(x, keep_prob, 17)
    score = tf.nn.softmax(output)
    f_cls = tf.argmax(score, 1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    saver.restore(sess, './model/model.ckpt-4999')
    for i in os.listdir(path):
        imgpath = os.path.join(path, i)
        im = cv2.imread(imgpath)
        im = cv2.resize(im, (224, 224))

        im = np.expand_dims(im, axis=0)

        pred, _score = sess.run([f_cls, score], feed_dict={x: im, keep_prob: 1.0})
        prob = round(np.max(_score))
        print("{} flowers class is {}, score: {} ".format(i, int(pred), prob))

    sess.close()


if __name__ == "main":
    path = './Example/'
    test(path)
