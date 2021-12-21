"""
Dataset is Kaggle intro to NLP dataset: tweets labeled as disaster/not disaster. Original source available
at Kaggle: https://www.kaggle.com/c/nlp-getting-started
"""

from helper_functions import create_tensorboard_callback, plot_loss_curves, compare_histories

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import pandas as pd
import random
from sklearn.model_selection import train_test_split

TRAIN_PATH: str = "datasets/train.csv"
TEST_PATH: str = "datasets/test.csv"


def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df_shuffled = train_df.sample(frac=1, random_state=42)
    # visualize_train_data(train_df_shuffled)

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                                train_df_shuffled["target"].to_numpy(),
                                                                                test_size=0.1,
                                                                                random_state=42)
    # print(len(train_sentences), len(train_labels), len(val_sentences), len(val_labels))
    # print(train_sentences[:10], train_labels[:10])

    return train_sentences, val_sentences, test_df["text"], train_labels, val_labels


def tokenize_text_dataset(train_sentences, val_sentences, test_sentences, max_vocab_len=10000):
    avg_sentence_len = round(sum([len(i.split()) for i in train_sentences])/len(train_sentences))
    print("Avg sentence length: {avg_sentence_len}")

    text_vectorizer = TextVectorization(max_tokens=max_vocab_len,  # how many words in final vocab, None means unlimited
                                        standardize="lower_and_strip_punctuation",
                                        split="whitespace",
                                        ngrams=None,
                                        output_mode="int",
                                        output_sequence_length=avg_sentence_len,  # max length of token sequence
                                        pad_to_max_tokens=False,
                                        )
    text_vectorizer.adapt(train_sentences)
    words_in_vocab = text_vectorizer.get_vocabulary()
    print("Vocab length: ", len(words_in_vocab))
    print("Top 5 unique words in vocabulary: ", words_in_vocab[:5])
    print("Bottom 5 unique words in vocabulary: ", words_in_vocab[-5:])
    sample_sentence = "There's a flood in my street"
    print(f"Vectorized sentence '{sample_sentence}'", text_vectorizer([sample_sentence]))

    random_sentence = random.choice(train_sentences)
    print(f"Vectorized sentence '{random_sentence}'", text_vectorizer([random_sentence]))


def visualize_train_data(train_df):
    # print(train_df.head)
    # print(train_df["text"][1])

    # dataset is relatively balanced between classes
    # if it were not, see https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    # print(train_df.target.value_counts())

    # visualize random training examples
    rand_index = random.randint(0, len(train_df)-5)
    for row in train_df[["text", "target"]][rand_index:rand_index+5].itertuples():
        _, text, target = row
        print(f"Target: {target}", "(disaster)" if target > 0 else "(not disaster)")
        print(f"Text: {text}")
        print("-----")

def run():
    print("nlp fundies")
    train_sentences, val_sentences, test_sentences, train_labels, val_labels = load_data()
    tokenize_text_dataset(train_sentences, val_sentences, test_sentences)


