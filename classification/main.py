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
    _y_train_pred = cross_val_predict(classifier, train_data, train_labels, cv=3)
    return confusion_matrix(train_labels, _y_train_pred), _y_train_pred


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


def calc_roc_auc(train_labels, y_scores):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(train_labels, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.show()


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis("off")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = fetch_train_test_split()
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # for now, simplify problem to detecting number 5 only
    # sot convert labels so only fives are true
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    # trained_classifier = train_SGD(X_train, y_train_5)

    some_digit = X_train[0]
    # plot_digit(some_digit)
    # print(trained_classifier.predict([some_digit]))

    # print(do_cross_validation(trained_classifier, X_train, y_train_5))
    # from sklearn.model_selection import cross_val_score
    # print(cross_val_score(trained_classifier, X_train, y_train_5, cv=3, scoring='accuracy'))

    # print(do_custom_cross_validation(trained_classifier, X_train, y_train_5))

    # print(calc_confusion_matrix(trained_classifier, X_train, y_train_5))

    # print(calc_precision_and_recall_and_f1(trained_classifier, X_train, y_train_5))

    # precisions, recalls, thresholds = calc_pr_curve(trained_classifier, X_train, y_train_5)
    # plot_pr_curve(precisions, recalls, thresholds)

    # see that as precision increases, recall will fall:
    # plot_precision_v_recall(precisions, recalls)
    # fpr, tpr, thresholds = calc_roc_curve(trained_classifier, X_train, y_train_5)
    # plot_roc_curve(fpr, tpr)

    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_curve

    forest_clf = RandomForestClassifier(random_state=42, n_estimators=10)
    y_probs_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
    y_scores_forest = y_probs_forest[:, 1]
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
    """

    """
    # multiclass classifier
    # SVC will automatically select OvO or OvR multiclass classifiers when needed
    # SVC Support Vector Machine classifier
    from sklearn.svm import SVC
    from sklearn.multiclass import OneVsRestClassifier
    svm_clf = SVC()
    svm_clf.fit(X_train, y_train)
    print(svm_clf.predict([some_digit]))
    some_digit_scores = svm_clf.decision_function([some_digit])
    print(some_digit_scores)

    # Force a OvR classifier with SVC
    over_clf = OneVsRestClassifier(SVC())
    over_clf.fit(X_train, y_train)
    print(over_clf.predict([some_digit]))
    print(len(over_clf.estimators_))
    """

    """
    # SGD model
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import SGDClassifier

    sgd_clf = SGDClassifier(random_state=42, max_iter=5, tol=-np.infty)
    sgd_clf.fit(X_train, y_train)
    print(sgd_clf.predict([some_digit]))
    print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy'))

    # now scale data to improve performance
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))

    # plot confusion matrix to analyze any errors
    conf_mx, y_train_pred = calc_confusion_matrix(sgd_clf, X_train_scaled, y_train)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()

    # normalize confusion matrix to get better view of errors only
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()

    # plot sample error digit images
    cl_a, cl_b = 3, 5
    X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
    X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
    X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
    X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plot_digits(X_aa[:25], images_per_row=5)
    plt.subplot(222)
    plot_digits(X_ab[:25], images_per_row=5)
    plt.subplot(223)
    plot_digits(X_ba[:25], images_per_row=5)
    plt.subplot(224)
    plot_digits(X_bb[:25], images_per_row=5)
    plt.show()
    """

    # multilabel classification
    # use same digits data, but output if the digit is larger than 7 and odd/even
    from sklearn.neighbors import KNeighborsClassifier
    y_train_large = (y_train >= 7)
    y_train_odd = (y_train % 2 == 1)
    y_multilabel = np.c_[y_train_large, y_train_odd]

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel)
    print(knn_clf.predict([some_digit]))

    from sklearn.metrics import f1_score
    from sklearn.model_selection import cross_val_predict

    # n_jobs=-1 means use all available cpus, cv=3 means 3 splits for cross val comparisons
    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1)
    print(f1_score(y_multilabel, y_train_knn_pred, average='macro'))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
