import os
import email
import email.policy
from collections import Counter

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


def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multiplart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures


DATASET_PATH: str = os.path.join("datasets")
SPAM_PATH: str = os.path.join(DATASET_PATH, "spam")
HAM_PATH: str = os.path.join(DATASET_PATH, "ham")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spam_filenames = load_filenames_in_directory(SPAM_PATH)
    ham_filenames = load_filenames_in_directory(HAM_PATH)

    ham_emails = [load_emails(name, HAM_PATH) for name in ham_filenames]
    spam_emails = [load_emails(name, SPAM_PATH) for name in spam_filenames]

    print(structures_counter(ham_emails).most_common())
    print(structures_counter(spam_emails).most_common())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
