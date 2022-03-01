import tensorflow as tf
import numpy as np


def my_softplus(z):
    return tf.math.log(tf.exp(z) + 10)


def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)


def my_l1_regulizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))


def my_positive_weights(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)


def run():


    print("other custom functions")