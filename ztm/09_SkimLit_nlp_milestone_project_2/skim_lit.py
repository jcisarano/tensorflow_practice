import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

from helper_functions import calculate_results


DATA_DIR_20K_NUM_REPL: str = "dataset/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/"
DATA_DIR_200K_NUM_REPL: str = "dataset/pubmed-rct-master/PubMed_200k_RCT_numbers_replaced_with_at_sign/"


def convert_to_panda_df(train_samples, val_samples, test_samples):
    train_df = pd.DataFrame(train_samples)
    val_df = pd.DataFrame(val_samples)
    test_df = pd.DataFrame(test_samples)

    return train_df, val_df, test_df


def get_lines(filepath):
    """
    Load text file and returns lines in that file as list

    :return:
        List of strings with one string per line from target file
    """
    with open(filepath, "r") as f:
        return f.readlines()


def preprocess_text_with_line_numbers(filepath):
    """
    Returns list of dictionaries of abstract line data. Sample data format:
        [{
            'line_number' : 0,
            'target': 'BACKGROUND',
            'text': 'Emotional eating is associated with overeating and the development of obesity .',
            'total_lines': 11
        },
        ...
        ]
    :param filepath:
    :return:
    """
    input_lines = get_lines(filepath)
    abstract_lines = ""
    abstract_samples = []

    for line in input_lines:
        if line.startswith("###"):  # this is an id line, the start of an abstract
            abstract_id = line
            abstract_lines = ""
        elif line.isspace():  # this is an empty line, the end of an abstract
            abstract_line_split = abstract_lines.splitlines()

            # now iterate through line in a single abstract to handle and count them:
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}
                target_text_split = abstract_line.split("\t")  # splits id from text
                line_data["target"] = target_text_split[0]
                line_data["text"] = target_text_split[1].lower()
                line_data["line_number"] = abstract_line_number  # what number is this line in the abstract?
                line_data["total_lines"] = len(abstract_line_split) - 1
                abstract_samples.append(line_data)
        else:
            abstract_lines += line

    return abstract_samples


def visualize_data(train_df, val_df, test_df):
    print(train_df.head(14))

    # distribution of labels
    print(train_df.target.value_counts())

    # line length
    train_df.total_lines.plot.hist()
    plt.show()

    train_sentences = train_df["text"].tolist()
    val_sentences = val_df["text"].tolist()
    test_sentences = test_df["text"].tolist()

    print(train_sentences[:10])


def examine_sentence_data(sentences):
    sent_lens = [len(sentence.split()) for sentence in sentences]
    avg_sent_len = np.mean(sent_lens)

    print("Average sentence length:", avg_sent_len)

    plt.hist(sent_lens, bins=30)
    plt.show()

    # determine 95th percentile sentence length
    # i.e. 95% of sentences are shorter than this length
    output_sequence_len = int(np.percentile(sent_lens, 95))
    print("95% of sentences are below length ", output_sequence_len)
    print("Max sentence length: ", max(sent_lens))
    longest = sentences[np.argmax(sent_lens)]
    print("Longest sentence:", longest)

def get_labels_one_hot(y_train, y_val, y_test):
    from sklearn.preprocessing import OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse=False)
    train_labels_one_hot = one_hot_encoder.fit_transform(y_train)
    val_labels_one_hot = one_hot_encoder.transform(y_val)
    test_labels_one_hot = one_hot_encoder.transform(y_test)

    return train_labels_one_hot, val_labels_one_hot, test_labels_one_hot


def get_labels_int_encode(y_train, y_val, y_test):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(y_train)
    val_labels_encoded = label_encoder.transform(y_val)
    test_labels_encoded = label_encoder.transform(y_test)

    return train_labels_encoded, val_labels_encoded, test_labels_encoded, label_encoder.classes_


def fit_naive_bayes(X_train, y_train, X_val, y_val):
    """
    Fit and train TF-IDF Multinomial Naive Bayes model as baseline for comparison to all other models.
    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :return:
    """

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    results = calculate_results(y_val, preds)

    return model, results


def parse_file(filepath):
    """
    My messed up version of the preprocess_text_with_line_numbers() task.
    :param filepath:
    :return:
    """
    lines = get_lines(filepath)
    parsed_lines = []
    for line in lines:
        line = line.replace("\n", "")
        if line.startswith("###"):
            abstract = {"line_number": 0, "total_lines": 0}
        elif len(line) == 0:
            parsed_lines.append(abstract)
            continue
        else:
            split = line.split("\t")
            abstract["target"] = split[0]
            abstract["text"] = split[1]
            abstract["total_lines"] += 1

    print(len(parsed_lines))
    print(parsed_lines[:5])

    return parsed_lines


def run():
    train_samples = preprocess_text_with_line_numbers(filepath=DATA_DIR_20K_NUM_REPL + "train.txt")
    val_samples = preprocess_text_with_line_numbers(filepath=DATA_DIR_20K_NUM_REPL + "dev.txt")
    test_samples = preprocess_text_with_line_numbers(filepath=DATA_DIR_20K_NUM_REPL + "test.txt")
    train_df, val_df, test_df = convert_to_panda_df(train_samples, val_samples, test_samples)
    # visualize_data(train_df, val_df, test_df)

    train_labels_one_hot, val_labels_one_hot, test_labels_one_hot = get_labels_one_hot(
        train_df["target"].to_numpy().reshape(-1, 1),
        val_df["target"].to_numpy().reshape(-1, 1),
        test_df["target"].to_numpy().reshape(-1, 1)
    )

    train_labels_encoded, val_labels_encoded, test_labels_encoded, class_names = get_labels_int_encode(
        train_df["target"].to_numpy(),
        val_df["target"].to_numpy(),
        test_df["target"].to_numpy(),
        )

    # model_0, model_0_results = fit_naive_bayes(train_df["text"], train_labels_encoded, val_df["text"], val_labels_encoded)
    # print(model_0_results)

    examine_sentence_data(train_df["text"].to_numpy())
