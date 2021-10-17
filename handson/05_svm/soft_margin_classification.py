
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

import utils


def plot_hard_sensitivity_examples(X, y):
    """
    SVMs with 'hard margins' enforce the idea that all instances must be off the street and only on the side matching
    their own class.
    These two plots show the limitations of hard margins.
    The one on the left shows that with an outlier mixed in with the data of another class, no margin can be found.
    The one on the right shows that an outlier very near the data of another class will create margins so narrow that
    they probably will not generalize well at all.
    :param X:
    :param y:
    :return:
    """
    X_outliers = np.array([[3.4, 1.3], [3.2, 0.8]])
    y_outliers = np.array([0, 0])
    Xo1 = np.concatenate([X, X_outliers[:1]], axis=0)
    yo1 = np.concatenate([y, y_outliers[:1]], axis=0)
    Xo2 = np.concatenate([X, X_outliers[1:]], axis=0)
    yo2 = np.concatenate([y, y_outliers[1:]], axis=0)

    svm_clf = SVC(kernel="linear", C=10**9)
    svm_clf.fit(Xo2, yo2)

    fig, axes = plt.subplots(ncols=2, figsize=(10, 2.7), sharey=True)

    plt.sca(axes[0])
    plt.plot(Xo1[:, 0][yo1 == 1], Xo1[:, 1][yo1 == 1], "bs")
    plt.plot(Xo1[:, 0][yo1 == 0], Xo1[:, 1][yo1 == 0], "yo")
    plt.text(0.3, 1.0, "Impossible!", fontsize=24, color="red")
    plt.xlabel("Petal length", fontsize=14)
    plt.xlabel("Petal width", fontsize=14)
    plt.annotate("Outlier",
                 xy=(X_outliers[0][0], X_outliers[0][1]),
                 xytext=(2.5, 1.7),
                 ha="center",
                 arrowprops=dict(facecolor="black", shrink=0.1),
                 fontsize=16,
                 )
    plt.axis([0, 5.5, 0, 2])

    plt.sca(axes[1])
    plt.plot(Xo2[:, 0][yo2 == 1], Xo2[:, 1][yo2 == 1], "bs")
    plt.plot(Xo2[:, 0][yo2 == 0], Xo2[:, 1][yo2 == 0], "yo")
    utils.plot_svc_decision_boundary(svm_clf=svm_clf, xmin=0, xmax=5.5)
    plt.xlabel("Petal length", fontsize=14)
    plt.annotate("Outlier",
                 xy=(X_outliers[1][0], X_outliers[1][1]),
                 xytext=(3.2, 0.08),
                 ha="center",
                 arrowprops=dict(facecolor="black", shrink=0.1),
                 fontsize=16,
                 )
    plt.axis([0, 5.5, 0, 2])
    plt.show()


#def plot_large_margin_vs_fewer_margin_violations():



def soft_margin_example():
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    X, y = utils.load_iris_data_width_length()
    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
    ])
    svm_clf.fit(X, y)
    return svm_clf


def soft_margin_svc_example():
    from sklearn.preprocessing import StandardScaler

    X, y = utils.load_iris_data_width_length()
    svc_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc_linear_kernel", SVC(kernel="linear", C=1)),
    ])

    svc_clf.fit(X, y)
    return svc_clf


def run():
    X, y = utils.load_iris_setosa_or_versicolor()

    plot_hard_sensitivity_examples(X, y)
    soft_marg_clf = soft_margin_example()
    print(soft_marg_clf.predict([[5.5, 1.7]]))

    soft_marg_svc_clf = soft_margin_svc_example()
    print(soft_marg_svc_clf.predict([[5.5, 1.7]]))
