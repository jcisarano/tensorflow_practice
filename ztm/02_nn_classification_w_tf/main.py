# Intro to NN classification with TensorFlow
# How to write NN for classification problems, where you need to classify something as one thing or another
# Types of classification:
# Binary classification
# Multiclass classification
# Multilabel classification

from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # make 1000 examples
    n_samples = 1000
    X, y = make_circles(n_samples,
                        noise=0.03,
                        random_state=42)
    # Check out features and labels:
    print(X[:10])
    print(y[:10])

    circles = pd.DataFrame({"X0:": X[:, 0], "X1": X[:, 1], "label": y})
    print(circles)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
