"""

"""
import tensorflow as tf


def run():
    set1 = tf.constant([[2, 3, 5, 7], [7, 9, 0, 0]])
    set2 = tf.constant([[4, 5, 6], [9, 10, 0]])

    # union of sets in last dimension
    print("\ntf.sparse.to_dense(tf.sets.union(set1, set2))\n",
          tf.sparse.to_dense(tf.sets.union(set1, set2)))

    # set difference of elements in last dimension. All but the last dimension of sets must match
    print("\ntf.sparse.to_dense(tf.sets.difference(set1, set2))\n",
          tf.sparse.to_dense(tf.sets.difference(set1, set2)))

    # Computes set intersection of elements in last dimension. All but last dimension of sets must match
    print("\ntf.sparse.to_dense(tf.sets.intersection(set1, set2))\n",
          tf.sparse.to_dense(tf.sets.intersection(set1, set2)))

    print("sets")
