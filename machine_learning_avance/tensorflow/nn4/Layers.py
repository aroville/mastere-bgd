

import tensorflow as tf
import numpy as np


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary(name + '/mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary(name + '/sttdev', stddev)
        tf.scalar_summary(name + '/max', tf.reduce_max(var))
        tf.scalar_summary(name + 'min/', tf.reduce_min(var))
        tf.histogram_summary(name, var)


def fc(tensor, output_dim, name, act=tf.nn.relu):
    with tf.name_scope(name):
        input_dim = tensor.get_shape()[1].value
        Winit = tf.truncated_normal([input_dim, output_dim], stddev=0.1)
        W = tf.Variable(Winit)
        print(name, 'input  ', tensor)
        print(name, 'W  ', W.get_shape())
        variable_summaries(W, name + '/W')
        Binit = tf.constant(0.0, shape=[output_dim])
        B = tf.Variable(Binit)
        variable_summaries(B, name + '/B')
        tensor = tf.matmul(tensor, W) + B
        tensor = act(tensor)
    return tensor


def conv(tensor, out_dim, filter_size, stride, name, act=tf.nn.relu):
    with tf.name_scope(name):
        in_dim_h = tensor.get_shape()[1].value
        in_dim_w = tensor.get_shape()[2].value
        in_dim_d = tensor.get_shape()[3].value
        w_init = tf.truncated_normal([filter_size, filter_size, in_dim_d, out_dim], stddev=0.1)
        w = tf.Variable(w_init)
        print(name, 'input  ', tensor)
        print(name, 'W  ', w.get_shape())
        variable_summaries(w, name + '/W')
        b_init = tf.constant(0.0, shape=[out_dim])
        b = tf.Variable(b_init)
        variable_summaries(b, name + '/B')
        tensor = tf.nn.conv2d(tensor, w, strides=[1, stride, stride, 1], padding='SAME') + b
        tensor = act(tensor)
    return tensor


def maxpool(tensor, pool_size, name):
    with tf.name_scope(name):
        tensor = tf.nn.max_pool(
            tensor,
            ksize=(1, pool_size, pool_size, 1),
            strides=(1, pool_size, pool_size, 1),
            padding='SAME')

    return tensor


def flat(tensor):
    in_dim_h = tensor.get_shape()[1].value
    in_dim_w = tensor.get_shape()[2].value
    in_dim_d = tensor.get_shape()[3].value
    tensor = tf.reshape(tensor, [-1, in_dim_h * in_dim_w * in_dim_d])
    print('flat output  ', tensor)
    return tensor


def unflat(tensor, out_dim_h, out_dim_w, out_dim_d):
    tensor = tf.reshape(tensor, [-1, out_dim_h, out_dim_w, out_dim_d])
    tf.image_summary('input', tensor, 10)
    print('unflat output  ', tensor)
    return tensor    