import os
import email
import email.policy
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import re
from html import unescape


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


def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub(r'<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)



DATASET_PATH: str = os.path.join("datasets")
SPAM_PATH: str = os.path.join(DATASET_PATH, "spam")
HAM_PATH: str = os.path.join(DATASET_PATH, "ham")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spam_filenames = load_filenames_in_directory(SPAM_PATH)
    ham_filenames = load_filenames_in_directory(HAM_PATH)

    ham_emails = [load_emails(name, HAM_PATH) for name in ham_filenames]
    spam_emails = [load_emails(name, SPAM_PATH) for name in spam_filenames]

    # look at structures/document types (e.g. text/plain, text/html, multipart, etc)
    print(structures_counter(ham_emails).most_common())
    print(structures_counter(spam_emails).most_common())

    # look at header info from one email
    for header, value in spam_emails[0].items():
        print(header, ":", value)

    # split into train/test data:
    # create X from combined array of ham and spam:
    X = np.array(ham_emails + spam_emails)
    # create labels by populating array with 0/1 since we know which are ham and spam:
    y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # strip out the HTML from the message bodies
    html_spam_emails = [email for email in X_train[y_train == 1]
                        if get_email_structure(email) == "text/html"]
    sample_html_spam = html_spam_emails[7]
    print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
