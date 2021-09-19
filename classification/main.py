import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


def plot_digit(data, size=28):
    data_img = data.reshape(size, size)
    plt.imshow(data_img, cmap="binary", interpolation="nearest")
    plt.axis("off")
    plt.show()


def fetch_train_test_split():
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    mnist.target = mnist.target.astype(np.int8)  # converts from string to int
    X, y = mnist["data"], mnist["target"]
    return X[:60000], X[60000:], y[:60000], y[60000:]


def train_SGD(X_train, y_train):
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(random_state=42, max_iter=5, tol=-np.infty)
    sgd_clf.fit(X_train, y_train)
    return sgd_clf


def do_cross_validation(classifier, train_data, train_labels, cv=3, scoring='accuracy'):
    from sklearn.model_selection import cross_val_score
    return cross_val_score(classifier, train_data, train_labels, cv=cv, scoring=scoring)


def do_custom_cross_validation(classifier, train_data, train_labels):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

    results = []
    for train_index, test_index in skfolds.split(train_data, train_labels):
        clone_clf = clone(classifier)
        X_train_folds = train_data[train_index]
        y_train_folds = train_labels[train_index]
        X_test_fold = train_data[test_index]
        y_test_fold = train_labels[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        results.append(n_correct / len(y_pred))

    return results


def calc_confusion_matrix(classifier, train_data, train_labels):
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    y_train_pred = cross_val_predict(classifier, train_data, train_labels, cv=3)
    return confusion_matrix(train_labels, y_train_pred)


def calc_precision_and_recall_and_f1(classifier, train_data, train_labels):
    from sklearn.metrics import precision_score, recall_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import f1_score
    y_train_pred = cross_val_predict(classifier, train_data, train_labels, cv=3)
    return precision_score(train_labels, y_train_pred), recall_score(train_labels, y_train_pred), f1_score(train_labels,
                                                                                                           y_train_pred)


def calc_pr_curve(classifier, train_data, train_labels):
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import precision_recall_curve
    y_train_pred = cross_val_predict(classifier, train_data, train_labels, cv=3, method="decision_function")
    return precision_recall_curve(train_labels, y_train_pred)


def plot_pr_curve(precisions, recalls, thresholds):
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
    plt.xlim([-700000, 700000])
    plt.show()


def plot_precision_v_recall(precisions, recalls):
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.show()


def calc_roc_curve(classifier, train_data, train_labels):
    from sklearn.metrics import roc_curve
    from sklearn.model_selection import cross_val_predict
    y_train_pred = cross_val_predict(classifier, train_data, train_labels, cv=3, method="decision_function")
    return roc_curve(train_labels, y_train_pred)


def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = fetch_train_test_split()
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # for now, simplify problem to detecting number 5 only
    # sot convert labels so only fives are true
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    trained_classifier = train_SGD(X_train, y_train_5)

    # some_digit = X_train[0]
    # plot_digit(some_digit)
    # print(trained_classifier.predict([some_digit]))

    print(do_cross_validation(trained_classifier, X_train, y_train_5))
    # from sklearn.model_selection import cross_val_score
    # print(cross_val_score(trained_classifier, X_train, y_train_5, cv=3, scoring='accuracy'))

    # print(do_custom_cross_validation(trained_classifier, X_train, y_train_5))

    # print(calc_confusion_matrix(trained_classifier, X_train, y_train_5))

    # print(calc_precision_and_recall_and_f1(trained_classifier, X_train, y_train_5))

    # precisions, recalls, thresholds = calc_pr_curve(trained_classifier, X_train, y_train_5)
    # plot_pr_curve(precisions, recalls, thresholds)

    # see that as precision increases, recall will fall:
    # plot_precision_v_recall(precisions, recalls)
    fpr, tpr, thresholds = calc_roc_curve(trained_classifier, X_train, y_train_5)
    plot_roc_curve(fpr, tpr)

"""
# sort_by_target(mnist) # not sure about this - the jupyter notebook says it is needed? but w/o, my results match the book
print(mnist["data"], mnist["target"])
print(mnist.keys())

X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

print(y[0])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
"""

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
