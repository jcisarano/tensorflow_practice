# introduction to regression with neural networks in tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print(tf.__version__)
    X = np.array([-7., -4., -1., 2., 5., 8., 11., 14.])

    y = np.array([3., 6., 9., 12., 15., 18., 21., 24.])

    plt.scatter(X, y, c="red")
    plt.show()

    # examining desired output shape
    house_info = tf.constant(["bedroom", "bathroom", "garage"])
    house_price = tf.constant([939700])
    print(house_info, house_price)

    input_shape = X[0].shape
    output_shape = y[0].shape
    print(input_shape, output_shape)
    print(X[0], y[0])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
