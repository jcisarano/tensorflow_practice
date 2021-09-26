import os

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


DATASET_PATH: str = os.path.join("datasets")
SPAM_PATH: str = os.path.join(DATASET_PATH, "spam")
HAM_PATH: str = os.path.join(DATASET_PATH, "ham")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spam_filenames = load_filenames_in_directory(SPAM_PATH)
    ham_filenames = load_filenames_in_directory(HAM_PATH)

print(len(spam_filenames))
print(len(ham_filenames))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
