import tensorflow as tf
import numpy as np


def run():
    print(tf.constant(b"hello world"))
    print(tf.constant("café"))
    u = tf.constant([ord(c) for c in "café"])  # The ord() function returns int representing the Unicode character
    print(u)

    b = tf.strings.unicode_encode(u, "UTF-8")
    print(b)  # same as original constant
    print("tf.strings.length(b)", tf.strings.length(b, unit="UTF8_CHAR"))
    print("tf.strings.unicode_decode(b)", tf.strings.unicode_decode(b, "UTF-8"))  # same as u

    p = tf.constant(["Café", "Coffee", "caffè", "咖啡"])
    print("tf.strings.length(p)", tf.strings.length(p, unit="UTF8_CHAR"))

    r = tf.strings.unicode_decode(p, "UTF8")
    print("tf.strings.unicode_decode(r)", r)

    print("strings")
