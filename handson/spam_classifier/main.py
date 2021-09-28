import os
import email
import email.policy
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import nltk
from scipy.sparse import csr_matrix

import re
from html import unescape
import urlextract


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


def email_to_text(email):
    html = None
    for part in email.walk():
        content_type = part.get_content_type()
        if content_type not in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except:  # in case of encoding issues
            content = str(part.get_payload())
        if content_type == "text/plain":
            return content
        else:
            html = content
        if html:
            return html_to_plain_text(html)


class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)


class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.most_common_ = most_common
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self

    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))


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
    # print(structures_counter(ham_emails).most_common())
    # print(structures_counter(spam_emails).most_common())

    # look at header info from one email
    # for header, value in spam_emails[0].items():
    #     print(header, ":", value)

    # split into train/test data:
    # create X from combined array of ham and spam:
    X = np.array(ham_emails + spam_emails)
    # create labels by populating array with 0/1 since we know which are ham and spam:
    y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # strip out the HTML from the message bodies
    html_spam_emails = [email for email in X_train[y_train == 1]
                        if get_email_structure(email) == "text/html"]
    # sample_html_spam = html_spam_emails[7]
    # print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")
    # print(email_to_text(sample_html_spam)[:100], "...")

    # test stemming
    stemmer = nltk.PorterStemmer()
    # for word in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):
    #    print(word, "->", stemmer.stem(word))

    # replace urls with word URL
    url_extractor = urlextract.URLExtract()
    # print(url_extractor.find_urls("Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"))

    # test EmailToWordCounterTransformer
    # X_sample = X_train[:3]
    # X_sample_wordcounts = EmailToWordCounterTransformer().fit_transform(X_sample)
    # print(X_sample_wordcounts)

    # test WordCounterToVectorTransformer
    # vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
    # sample_X_vectors = vocab_transformer.fit_transform(X_sample_wordcounts)
    # print(sample_X_vectors.toarray())

    preprocess_pipeline = Pipeline([
        ("email_to_wordcount", EmailToWordCounterTransformer()),
        ("wordcount_to_vector", WordCounterToVectorTransformer())
    ])

    X_train_transformed = preprocess_pipeline.fit_transform(X_train)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
