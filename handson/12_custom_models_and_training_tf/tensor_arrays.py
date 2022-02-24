"""

"""
import tensorflow as tf


def run():
    array = tf.TensorArray(dtype=tf.float32, size=3)
    array = array.write(0, tf.constant([1., 2.]))
    array = array.write(1, tf.constant([3., 10.]))
    array = array.write(2, tf.constant([5., 7.]))
    print("\narray.read(1)\n", array.read(1))

    print("\narray.stack()\n", array.stack())

    mean, variance = tf.nn.moments(array.stack(), axes=0)
    print("\nmean\n", mean)
    print("\nvariance\n", variance)
    print("tensor arrays")
