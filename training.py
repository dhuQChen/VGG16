import tensorflow as tf
import numpy as np
from datetime import datetime
from VGG16 import *
import matplotlib.pyplot as plt

batch_size = 32
lr = 0.00001
n_classes = 17
max_steps = 500


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int64)
    return img, label


def train():
    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
    y = tf.placeholder(tf.int64, shape=[None, n_classes], name='label')
    keep_prob = tf.placeholder(tf.float32)
    output = VGG16(x, keep_prob, n_classes)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))

    train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), tf.float32))

    images, labels = read_and_decode('train.tfrecords')
    img_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                    batch_size=batch_size,
                                                    capacity=512,
                                                    min_after_dequeue=200)
    label_batch = tf.one_hot(label_batch, n_classes, 1, 0)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    plot_loss = []
    fig = plt.figure()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(max_steps):
            batch_x, batch_y = sess.run([img_batch, label_batch])
            _, loss_val = sess.run([train_step, loss], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            if i % 100 == 0:
                train_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                print("%s: Step [%d] Loss: %f, training accuracy: %g" % (datetime.now(), i, loss_val, train_acc))

            plot_loss.append(loss_val)

            if (i + 1) == max_steps:
                saver.save(sess, './model/model.ckpt', global_step=i)

        coord.request_stop()
        coord.join(threads)

        plt.plot(plot_loss, c='r')
        plt.show()


if __name__ == '__main__':
    train()
