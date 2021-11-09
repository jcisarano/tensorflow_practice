import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def fetch_mnist_data(size=70000):
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    mnist.target = mnist.target.astype(np.int8)  # converts from string to int
    X, y = mnist["data"], mnist["target"]
    if size < len(X):
        X = X[:size]
        y = y[:size]
    return X, y


def train_random_forest(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
    clf.fit(X_train, y_train)
    print("Random forest score", clf.score(X_test, y_test))

    return clf


def train_svm(X_train, y_train, X_test, y_test):
    clf = SVC()
    clf.fit(X_train, y_train)
    print("SVM score:", clf.score(X_test, y_test))

    return clf


def train_extra_trees(X_train, y_train, X_test, y_test):
    clf = ExtraTreesClassifier()
    clf.fit(X_train, y_train)
    print("Extra trees score:", clf.score(X_test, y_test))

    return clf


def train_val_test_split(X, y, train_size=0.714, test_size=0.143, validation_size=0.143):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=test_size/(test_size+validation_size),
                                                    shuffle=False)
    return X_train, X_test, X_val, y_train, y_test, y_val


def run():
    X, y = fetch_mnist_data()
    X_train, X_test, X_val, y_train, y_test, y_val = train_val_test_split(X, y)
    print(X_train.shape, X_test.shape, X_val.shape)

    forest_clf = train_random_forest(X_train, y_train, X_val, y_val)
    etrees_clf = train_extra_trees(X_train, y_train, X_val, y_val)
    svm_clf = train_svm(X_train, y_train, X_val, y_val)

    # voting_clf = VotingClassifier(
    #     estimators=[("rf", forest_clf), ("et", etrees_clf), ("svc", svm_clf)],
    #     voting="hard"
    # )

    # voting_clf.fit(X_train, y_train)
    # print("Voting score:", voting_clf.score(X_test, y_test))

    # now, build a blender classifier based on predictions from each model
    rf_pred = forest_clf.predict(X_val)
    etrees_pred = etrees_clf.predict(X_val)
    svm_pred = svm_clf.predict(X_val)

    pred_set = np.stack([rf_pred, etrees_pred, svm_pred], axis=1)
    print(pred_set.shape)

    blender_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
    blender_clf.fit(pred_set, y_val)

    estimators = [forest_clf, etrees_clf, svm_clf]
    # now make a prediction using blender based on test set
    X_test_predictions = np.empty((len(y_test), len(estimators)), dtype=np.float32)

    # capture the predictions of the various classifiers
    for index, estimator in enumerate(estimators):
        X_test_predictions[:, index] = estimator.predict(X_test)

    # then make a prediction using them as the dataset
    y_pred = blender_clf.predict(X_test_predictions)
    print(accuracy_score(y_test, y_pred))

