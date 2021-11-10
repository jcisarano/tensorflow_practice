"""
Dimensionality reduction attempts to reduce the number of features in a dataset to speed training
and also make it easier to find a good solution to the problem.

E.g. edge pixels in some kinds of image recognition might not be important, or adjacent pixels might be redundant.

It can also make data visualization easier in some cases by reducing the number of dimensions down to two or three.

Dimensionality reduction can result in some data loss, so it can be a tradeoff between speed and accuracy.
It can also add complexity to the production pipeline.
"""


import data_utils as du

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(du.get_3d_dataset().shape)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
