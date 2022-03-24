"""
Given a character, or sequence of characters, what is the most probable next character?

Uses GRU, but could be swapped out with LSTM
"""

import tensorflow as tf

import numpy as np
import os
import time

PATH_TO_FILE: str = "datasets/nlp/shakespeare.txt"


def load_data():
    text = open(PATH_TO_FILE, "rb").read().decode(encoding="utf-8")
    vocab = sorted(set(text))

    print(f"Length of text: {len(text)} characters")
    print(text[:250])
    print(f"{len(vocab)} unique chars")
    print(vocab)

    return text, vocab


def text_from_ids(chars_from_ids: tf.keras.layers.StringLookup, ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


def split_input_target(sequence):
    """
    Takes a sequence as input, duplictes it, and shifts it to align the input and label for each timestep.
    For any input (letter) in the sequence, the label is the next letter in the sequence
    :param sequence:
    :return:
    """
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


def run():
    text, vocab = load_data()

    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None
    )
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
    )

    all_ids = ids_from_chars(tf.strings.unicode_split(text, "UTF-8"))
    print(all_ids)
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    for ids in ids_dataset.take(10):
        print(chars_from_ids(ids).numpy().decode("utf-8"))

    seq_length = 100
    examples_per_epoch = len(text) // (seq_length + 1)
    sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

    for seq in sequences.take(1):
        print(chars_from_ids(seq))

    for seq in sequences.take(5):
        print(text_from_ids(chars_from_ids, seq).numpy())

    dataset = sequences.map(split_input_target)

    for input_example, target_example in dataset.take(1):
        print("Input :", text_from_ids(chars_from_ids, input_example).numpy())
        print("Target :", text_from_ids(chars_from_ids, target_example).numpy())

    # Create training batches
    BATCH_SIZE: int = 64

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE: int = 64
    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024

    model = MyModel(
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units
    )

    # see example predictions
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size")

    # output as numbers
    print(model.summary())
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    print(sampled_indices)

    # convert to text
    print("Input:\n", text_from_ids(chars_from_ids, input_example_batch[0]).numpy())
    print("\nNext Char Predictions:\n", text_from_ids(chars_from_ids, sampled_indices).numpy())

    print("nlp text gen")

