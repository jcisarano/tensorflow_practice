from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import time


def get_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    mnist.target = mnist.target.astype(np.uint8)

    X = mnist["data"]
    y = mnist["target"]

    return train_test_split(X, y, train_size=60000)


def run():
    X_train, X_test, y_train, y_test = get_mnist()

    print(X_train.shape, X_test.shape)
    # rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # t1 = time.time()
    # rnd_clf.fit(X_train, y_train)
    # t2 = time.time()
    # y_pred_rf = rnd_clf.predict(X_test)
    # print("Random forest score:", rnd_clf.score(X_test, y_test))
    # print("Random forest accuracy score:", accuracy_score(y_test, y_pred_rf))
    # print("Random forest time:", t2 - t1)

    pca = PCA(n_components=0.95)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)

    # random forest doesn't get better with PCA on this dataset
    # it trains slower, and the accuracy is worse
    # rnd_clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
    # t1 = time.time()
    # rnd_clf_pca.fit(X_train_reduced, y_train)
    # t2 = time.time()
    # y_pred_rf_pca = rnd_clf_pca.predict(X_test_reduced)
    # print("RF PCA score:", rnd_clf_pca.score(X_test_reduced, y_test))
    # print("RF PCA accuracy:", accuracy_score(y_test, y_pred_rf_pca))
    # print("RF PCA time:", t2 - t1)

    # try softmax regression
    log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
    t1 = time.time()
    log_clf.fit(X_train, y_train)
    t2 = time.time()
    y_pred_lr = log_clf.predict(X_test)

    print("LR score:", log_clf.score(X_test, y_test))
    print("LR accuracy score:", accuracy_score(y_test, y_pred_lr))
    print("LR training time:", t2 - t1)




