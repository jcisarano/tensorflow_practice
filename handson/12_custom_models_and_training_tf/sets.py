"""

"""
import tensorflow as tf


def run():
    set1 = tf.constant([[2, 3, 5, 7], [7, 9, 0, 0]])
    set2 = tf.constant([[4, 5, 6], [9, 10, 0]])
    print("\ntf.sparse.to_dense(tf.sets.union(set1, set2))\n",
          tf.sparse.to_dense(tf.sets.union(set1, set2)))

    print("\ntf.sparse.to_dense(tf.sets.difference(set1, set2))\n",
          tf.sparse.to_dense(tf.sets.difference(set1, set2)))

    print("\ntf.sparse.to_dense(tf.sets.intersection(set1, set2))\n",
          tf.sparse.to_dense(tf.sets.intersection(set1, set2)))

    print("sets")
