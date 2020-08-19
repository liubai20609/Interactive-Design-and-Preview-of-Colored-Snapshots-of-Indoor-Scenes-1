# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number from 0 to 9:
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    bises = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + bises

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs


# ¼ÆËã×¼È·¶È
def compute_accuracy(x, y):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: x})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: x, ys: y})
    return result


# def placeholder for inputs
xs = tf.placeholder(tf.float32, [None, 784])  # 28*28
ys = tf.placeholder(tf.float32, [None, 10])  # 10¸öÊä³ö

# add output layer
prediction = add_layer(xs, 784, 10, tf.nn.softmax)  # softmax³£ÓÃÓÚ·ÖÀà

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 100 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
    # plt.scatter(batch_xs, batch_xs, c='r', alpha=0.5, label='准确率')
    # plt.xlabel('迭代次数')
    # plt.ylabel('准确率')
    # plt.legend()
    # plt.show()
