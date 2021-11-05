import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def get_moons():
    X, y = get_raw_moons()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test


def get_raw_moons():
    return make_moons(n_samples=500, noise=0.30, random_state=42)


def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.45, -1, 1.5], alpha=0.5, contour=True):
    # np.linspace(x,y,z) creates z evenly spaced values over given range (x,y)
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    # np.meshgrid creates regular xy grid with all combinations of x1s, x2s
    # here shape is (100,100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]

    # make predictions based on regular data generated above
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])
    # plt.contourf fills background color, plt.contour does not
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)

    # draws a heavier line over the prediction boundary for contrast
    if contour:
        custom_cmap2 = ListedColormap(["#7d7d58", "#4c4c7f", "#507d50"])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)

    # plots original data for reference
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", alpha=alpha)

    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)


def plot_decision_tree_v_bagging(X_train, X_test, y_train, y_test):
    # bagging classifier creates 500 models, each trained on only 100 random instances from the training set
    # uses soft voting by default, to switch from bagging to pasting (no replacement) set bootstrap=False
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        max_samples=100, bootstrap=True, random_state=42,
        n_jobs=-1
    )
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)
    print("BaggingClassifier accuracy", accuracy_score(y_test, y_pred))

    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred_tree = tree_clf.predict(X_test)
    print("DecisionTreeClassifier accuracy:", accuracy_score(y_test, y_pred_tree))

    X, y = get_raw_moons()
    fix, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.sca(axes[0])
    plot_decision_boundary(tree_clf, X, y)
    plt.title("Decision Tree", fontsize=14)
    plt.sca(axes[1])
    plot_decision_boundary(bag_clf, X, y)
    plt.title("Decision Trees with Bagging", fontsize=14)
    plt.ylabel("")
    plt.show()


def oob_classifier(X, y, X_test, y_test):
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        bootstrap=True, oob_score=True, random_state=40,
        n_jobs=-1
    )
    bag_clf.fit(X, y)
    print("OOB score:", bag_clf.oob_score_)
    # print(bag_clf.oob_decision_function_)  # returns the class probabilities for each training instance
    y_pred = bag_clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))


def run():
    X_train, X_test, y_train, y_test = get_moons()
    plot_decision_tree_v_bagging(X_train, X_test, y_train, y_test)

    """
    Bagging v pasting notes:
    Bagging subsets have more diversity than pasting, so there is slightly more bias
    But the extra diversity means the models are less correlated, so variance is reduced.
    Bagging is usually preferred because the models are better, but it can be worth doing
    cross-validation to evaluate both bagging and pasting to pick which one is better in 
    a given situation.
    """

    """
    Out-of-bag evaluation: There are some training instances that the predictor will not see,
    so these can be used as a validation set, without the need to create a separate set.
    sklearn has this built in, by setting oob_score=True
    """

    oob_classifier(X_train, y_train, X_test, y_test)

