import tensorflow as tf
import numpy as np


def run():
    p = tf.constant(["Café", "Coffee", "caffè", "咖啡"])
    r = tf.strings.unicode_decode(p, "UTF8")

    print("r[1]\n", r[1])
    print("\nr[1:3]\n", r[1:3])  # top index is exclusive, so this outputs r[1] and r[2]

    r2 = tf.ragged.constant([[65, 66], [], [67]])
    print("\ntf.concat([r, r2])\n", tf.concat([r, r2], axis=0))

    r3 = tf.ragged.constant([[68, 69, 70], [71], [], [72, 73]])
    print("\ntf.concat([r, r3])\n", tf.concat([r, r3], axis=0))

    print("\ntf.strings.unicode_encode(r3)\n", tf.strings.unicode_encode(r3, "UTF-8"))
    print("\nr.to_tensor()\n", r.to_tensor())

    print("ragged tensors")
