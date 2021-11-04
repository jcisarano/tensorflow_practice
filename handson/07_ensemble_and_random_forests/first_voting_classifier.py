from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_moons():
    """
    Get train/test data using moons dataset
    :return:
    """
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test


def run():
    X_train, X_test, y_train, y_test = get_moons()

    # individual classifiers:
    log_clf = LogisticRegression(solver="lbfgs", random_state=42)
    rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_clf = SVC(gamma="scale", random_state=42)

    # voting classifier combines each of the individuals
    # voting=hard makes this a hard classifier, the class with the most votes wins:
    voting_clf = VotingClassifier(
        estimators=[("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf)],
        voting="hard"
    )
    # fits them all together:
    voting_clf.fit(X_train, y_train)

    # write the accuracy score of each individual followed by the combined voting classifier
    # the voting classifier should win:
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
