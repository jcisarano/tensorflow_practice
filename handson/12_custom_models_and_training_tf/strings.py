import tensorflow as tf
import numpy as np


def run():
    print(tf.constant(b"hello world"))
    print(tf.constant("café"))
    u = tf.constant([ord(c) for c in "café"])  # The ord() function returns int representing the Unicode character
    print(u)

    b = tf.strings.unicode_encode(u, "UTF-8")
    print(tf.strings.length(b, unit="UTF8_CHAR"))
    print(tf.strings.unicode_decode(b, "UTF-8"))

    print("strings")
