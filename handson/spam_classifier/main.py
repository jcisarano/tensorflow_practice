import os
import email
import email.policy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# spam classifier
def load_filenames_in_directory(path):
    f = []
    for (_, _, filenames) in os.walk(path):
        f.extend(filenames)
    return f


def load_emails(filename, path):
    with open(os.path.join(path, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


DATASET_PATH: str = os.path.join("datasets")
SPAM_PATH: str = os.path.join(DATASET_PATH, "spam")
HAM_PATH: str = os.path.join(DATASET_PATH, "ham")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spam_filenames = load_filenames_in_directory(SPAM_PATH)
    ham_filenames = load_filenames_in_directory(HAM_PATH)

    print(len(ham_filenames))

    ham_emails = [load_emails(name, HAM_PATH) for name in ham_filenames]
    spam_emails = [load_emails(name, SPAM_PATH) for name in spam_filenames]


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
