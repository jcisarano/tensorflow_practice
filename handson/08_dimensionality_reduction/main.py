"""
Dimensionality reduction attempts to reduce the number of features in a dataset to speed training
and also make it easier to find a good solution to the problem.

E.g. edge pixels in some kinds of image recognition might not be important, or adjacent pixels might be redundant.

It can also make data visualization easier in some cases by reducing the number of dimensions down to two or three.

Dimensionality reduction can result in some data loss, so it can be a tradeoff between speed and accuracy.
It can also add complexity to the production pipeline.
"""

"""
The curse of dimensionality: High-dimensional datasets are at risk of being very sparse, i.e. all the points are very 
away from one another (see pp 215-216 for explanation). Since points are far apart, it is difficult to make good 
extrapolations in lower dimensions, meaning there is a high risk of overfitting.

One solution to this problem is to add more data to increase its density, but this is impractical/impossible even with
datasets of only 100 features.
"""

"""
Projection - Means of reducing dimensionality by projecting data onto a lower-dimension representation, e.g. project
data in three dimensions onto a 2d plane. (see p.216)

However, simple projection is not always the best approach, e.g. "Swiss roll" sample dataset where data wraps around in
3d space. A projection of this data would flatten it and mix layers together. A better method would be to _unwrap_ the
roll data. Or another way to flatten it: http://people.cs.uchicago.edu/~dinoj/manifold/swissroll.html

The goal is to turn the higher dimensional data into a good representation in 2d, which is known as a manifold.
"""

"""
Manifold - A lower dimensional representation that can be twisted into higher dimensional space. So a 2d manifold can be 
bent and twisted back into a 3d shape of its data.

Or more generally: a d-dimensional manifold is a part of an n-dimensional space (where d < n) that locally resembles a
d-dimensional hyperplane. The Swiss roll manifold locally resembles a 2d plane, but is rolled into the third dimension.

It is important to note that the decision boundaries on the lower-dimension manifold are not always simpler than those
in the higher-dimensional original representation--it will depend on the dataset and the problem...
"""


import data_utils as du

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(du.get_3d_dataset().shape)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
