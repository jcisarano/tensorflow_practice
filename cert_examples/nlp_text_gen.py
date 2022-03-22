"""
Given a character, or sequence of characters, what is the most probable next character?
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

    print("nlp text gen")
