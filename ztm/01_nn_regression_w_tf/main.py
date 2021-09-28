# introduction to regression with neural networks in tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print(tf.__version__)
    X = np.array([-7., -4., -1., 2., 5., 8., 11., 14.])

    y = np.array([3., 6., 9., 12., 15., 18., 21., 24.])

    plt.scatter(X, y)
    plt.show()

    

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
