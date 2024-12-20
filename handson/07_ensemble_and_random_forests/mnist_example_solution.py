import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier


def get_data():
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    mnist.target = mnist.target.astype(np.int8)  # converts from string to int
    X_train_val, X_test, y_train_val, y_test = train_test_split(mnist.data,
                                                                mnist.target,
                                                                test_size=10000,
                                                                random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)

    return X_train, X_test, X_val, y_train, y_val, y_test


def run():
    X_train, X_test, X_val, y_train, y_val, y_test = get_data()
    print(X_train.shape, X_test.shape)

    random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    svm_clf = LinearSVC(max_iter=100, tol=20, random_state=42)
    mlp_clf = MLPClassifier(random_state=42)

    # estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
    # for estimator in estimators:
    #     print("Training", estimator)
    #     estimator.fit(X_train, y_train)

    # print([estimator.score(X_val, y_val) for estimator in estimators])

    named_estimators = [
        ("random_forest_clf", random_forest_clf),
        ("extra_trees_clf", extra_trees_clf),
        ("svm_clf", svm_clf),
        ("mlp_clf", mlp_clf),
    ]

    voting_clf = VotingClassifier(named_estimators)
    print("Training", voting_clf)
    voting_clf.fit(X_train, y_train)
    print("Voting classifier score:", voting_clf.score(X_val, y_val))

    # remove SVM because it is the weakest to see if overall performance is improved
    voting_clf.set_params(svm_clf=None)
    print(voting_clf.estimators_)
    del voting_clf.estimators_[2]
    print(voting_clf.estimators_)
    print("Voting classifier score w/o SVM:", voting_clf.score(X_val, y_val))

    # test with soft voting:
    voting_clf.voting = "soft"
    print("Voting classifier soft voting score:", voting_clf.score(X_val, y_val))

    # switch back to hard voting, try it with test set:
    voting_clf.voting = "hard"
    print("Voting classifier test set score", voting_clf.score(X_test, y_test))

    print("Other classifiers test set scores")
    print([estimator.score(X_test, y_test) for estimator in voting_clf.estimators_])



