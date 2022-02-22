"""
SparseTensor is an efficient means of storing and processing a tensor that contains lots of zero values. TensorFlow
uses three dense tensors internally to represent the sparse tensor: indices, values, and dense_shape.
    values - A 1D tensor that lists the nonzero values of the sparse tensor
    indices - Specifies where in the sparse tensor the nonzero values appear
    dense_shape - Specifies the dimensions of the dense version of the sparse tensor
"""

import tensorflow as tf
import numpy as np


def run():
    s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],  # these are the locations of the nonzero values
                        values=[1., 2., 3.],  # these are the nonzero values
                        dense_shape=[3, 4])  # these are the dimensions of the dense version
    print(s)
    print("\ntf.sparse.to_dense(s)\n", tf.sparse.to_dense(s))

    s2 = s * 2.0
    print("\ns2\n", s2)

    try:
        s3 = s + 1
    except TypeError as ex:
        print(ex)

    s4 = tf.constant([[10., 20.], [30., 40.], [50., 60.], [70., 80.]])
    print("\ntf.sparse.sparse_dense_matmul(s, s4)\n", tf.sparse.sparse_dense_matmul(s, s4))

    s5 = tf.SparseTensor(indices=[[0, 2], [0, 1]],  # out of order
                         values=[1., 2.],
                         dense_shape=[3, 4])
    print("\ns5\n", s5)

    try:
        tf.sparse.to_dense(s5)
    except tf.errors.InvalidArgumentError as ex:
        print("\ntf.sparse.to_dense(s5) error\n", ex)

    s6 = tf.sparse.reorder(s5)  # fixes the order
    print("\ntf.sparse.to_dense(s6)\n", tf.sparse.to_dense(s6))

    print("sparse tensors")
