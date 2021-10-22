# Exercise: Train an SVM classifier on the MNIST dataset.
# Since SVM classifiers are binary classifiers, you will need to use one-versus-all to classify all 10 digits.
# You may want to tune the hyperparameters using small validation sets to speed up the process. What accuracy can you reach?


import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits, fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from scipy.stats import reciprocal, uniform
from sklearn.model_selection import RandomizedSearchCV

def run():
    digits = load_digits(as_frame=False)
    digits = fetch_openml("mnist_784", version=1, cache=True, as_frame=False)
    # _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    # for ax, image, label in zip(axes, digits["data"], digits["target"]):
    #     ax.set_axis_off()
    #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    #     ax.set_title("Training: %i" % label)
    # plt.show()

    X = digits["data"]
    y = digits["target"]

    test_idx = 60000
    X_train = X[:test_idx]
    X_test = X[test_idx:]
    y_train = y[:test_idx]
    y_test = y[test_idx:]

    # svc_clf = LinearSVC(random_state=42)
    # svc_clf.fit(X_train, y_train)

    # start out by checking agains training set (not test yet)
    # y_pred_train = svc_clf.predict(X_train)
    # print(accuracy_score(y_train, y_pred_train))

    # try scaling the data first
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
    X_test_scaled = scaler.fit_transform(X_test.astype(np.float32))

    # svc_clf = LinearSVC(random_state=42, verbose=1)
    # svc_clf.fit(X_train_scaled, y_train)
    # y_pred_train_scaled = svc_clf.predict(X_train_scaled)
    # print(accuracy_score(y_train,y_pred_train_scaled))

    svc = SVC(kernel="rbf", gamma="scale")
    # svc.fit(X_train_scaled[:10000], y_train[:10000])
    # y_pred = svc.predict(X_train_scaled)
    # print(accuracy_score(y_train, y_pred))

    params = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
    rnd_search_cv = RandomizedSearchCV(svc, params, n_iter=10, verbose=2, cv=3, n_jobs=-1)
    rnd_search_cv.fit(X_train_scaled[:1000], y_train[:1000])

    print(rnd_search_cv.best_estimator_)
    print(rnd_search_cv.best_score_)

    # now train best estimator on whole training set, do predictions, and check the results
    rnd_search_cv.best_estimator_.fit(X_train_scaled, y_train)
    y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
    print(accuracy_score(y_train, y_pred))



