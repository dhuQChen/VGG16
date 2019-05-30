import tensorflow as tf
import numpy as np

data_dict = np.load('vgg16.npy', allow_pickle=True, encoding='latin1').item()


def print_layer(t):
    print(t.op.name, '  ', t.get_shape().as_list(), '\n')


"""
权重初始化定义了3种方式：
    1.预训练模型参数
    2.截尾正态
    3.xavier
通过参数finetrun和xavier控制选择哪种方式
"""


def conv(x, out_channel, name, finetune=False):
    in_channel = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if finetune:
            weight = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
            print("finetune")
        else:
            weight = tf.Variable(tf.truncated_normal([3, 3, in_channel, out_channel], stddev=0.1), name="weights")
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[out_channel]), trainable=True, name="bias")
            print("truncated normal")

        conv = tf.nn.conv2d(x, weight, [1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(conv + bias, name=scope)
        print_layer(activation)
        return activation


def maxpool(x, name):
    activation = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name=name)
    print(activation)
    return activation


def fc(x, out_channel, name, finetune=False):
    in_channel = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if finetune:
            weight = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
            print("finetune")
        else:
            weight = tf.Variable(tf.truncated_normal([in_channel, out_channel], stddev=0.1), name="weights")
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[out_channel]), trainable=True, name="bias")
            print("truncated normal")

        # activation = tf.nn.relu_layer(x, weight, bias, name=name)
        # print_layer(activation)
        # return activation
        net = tf.add(tf.matmul(x, weight), bias)
        return net


def VGG16(images, _dropout, n_classes):
    # conv1
    conv1_1 = conv(images, 64, 'conv1_1', finetune=True)
    conv1_2 = conv(conv1_1, 64, 'conv1_2', finetune=True)
    pool1 = maxpool(conv1_2, 'pool1')

    # conv2
    conv2_1 = conv(pool1, 128, 'conv2_1', finetune=True)
    conv2_2 = conv(conv2_1, 128, 'conv2_2', finetune=True)
    pool2 = maxpool(conv2_2, 'pool2')

    # conv3
    conv3_1 = conv(pool2, 256, 'conv3_1', finetune=True)
    conv3_2 = conv(conv3_1, 256, 'conv3_2', finetune=True)
    conv3_3 = conv(conv3_2, 256, 'conv3_3', finetune=True)
    pool3 = maxpool(conv3_3, 'pool3')

    # conv4
    conv4_1 = conv(pool3, 512, 'conv4_1', finetune=True)
    conv4_2 = conv(conv4_1, 512, 'conv4_2', finetune=True)
    conv4_3 = conv(conv4_2, 512, 'conv4_3', finetune=True)
    pool4 = maxpool(conv4_3, 'pool4')

    # conv5
    conv5_1 = conv(pool4, 512, 'conv5_1', finetune=True)
    conv5_2 = conv(conv5_1, 512, 'conv5_2', finetune=True)
    conv5_3 = conv(conv5_2, 512, 'conv5_3', finetune=True)
    pool5 = maxpool(conv5_3, 'pool5')

    # fully connected layer
    flatten = tf.reshape(pool5, [-1, 7 * 7 * 512])
    fc_6 = fc(flatten, 4096, 'fc_6', finetune=False)
    fc_6 = tf.nn.relu(fc_6)
    dropout1 = tf.nn.dropout(fc_6, _dropout)

    fc_7 = fc(dropout1, 4096, 'fc_7', finetune=False)
    fc_7 = tf.nn.relu(fc_7)
    dropout2 = tf.nn.dropout(fc_7, _dropout)

    fc_8 = fc(dropout2, n_classes, 'fc_8', finetune=False)
    return fc_8






