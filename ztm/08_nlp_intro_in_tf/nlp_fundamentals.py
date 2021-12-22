"""
Dataset is Kaggle intro to NLP dataset: tweets labeled as disaster/not disaster. Original source available
at Kaggle: https://www.kaggle.com/c/nlp-getting-started
"""

from helper_functions import create_tensorboard_callback, plot_loss_curves, compare_histories

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
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
    # print("Vocab length: ", len(words_in_vocab))
    # print("Top 5 unique words in vocabulary: ", words_in_vocab[:5])
    # print("Bottom 5 unique words in vocabulary: ", words_in_vocab[-5:])
    # sample_sentence = "There's a flood in my street"
    # print(f"Vectorized sentence '{sample_sentence}'", text_vectorizer([sample_sentence]))

    # random_sentence = random.choice(train_sentences)
    # print(f"Vectorized sentence '{random_sentence}'", text_vectorizer([random_sentence]))

    return text_vectorizer


def create_embedding_for_text_dataset(train_sentences, val_sentences, test_sentences, max_vocab_len=10000):
    """
    Convert positive integers (indexes) into dense vector of fixed size.
    Parameters we care most about for embedding layer:
    `input_dim` - size of vocabulary
    `output_dim` - size of output embedding vector, e.g. a value of 100 means each token is represented by vector of 100
    `input_length` - length of sequences passed to embedding layer
    :return:
    """
    embedding = tf.keras.layers.Embedding(input_dim=max_vocab_len,
                                          output_dim=128,
                                          embeddings_initializer="uniform",
                                          input_length=len(train_sentences)
                                          )
    text_vectorizer = tokenize_text_dataset(train_sentences, val_sentences, test_sentences)
    # print(embedding)
    # random_sentence = random.choice(train_sentences)
    # print(f"Original sentence: {random_sentence}, Embedded version:", embedding(text_vectorizer([random_sentence])))

    return embedding


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
    # load the data:
    train_sentences, val_sentences, test_sentences, train_labels, val_labels = load_data()

    # Convert the text data to numbers because NN cannot handle text
    # Make sure the tensors are the right shape (may need to pad sequences to a standard length)
    # tokenize_text_dataset(train_sentences, val_sentences, test_sentences)
    create_embedding_for_text_dataset(train_sentences, val_sentences, test_sentences)

    """
    Series of modeling experiments
    Model 0: Naive Bayes will be the baseline
    Model 1: Feed-Forward neural network (dense model)
    Model 2: LSTM (RNN)
    Model 3: GRU (RNN)
    Model 4: Bidirectional-LSTM (RNN)
    Model 5: 1D Convolutional Neural Network (CNN)
    Model 6: TensorFlow Hub Pretrained Feature Extractor (transfer learning for NLP)
    Model 7: Same as model 6 with 10% of training data
    
    Standard tensorflow modeling steps:
    1) Create a model
    2) Build the model
    3) Fit the model
    4) Evaluate the model
    """


