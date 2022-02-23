"""

"""
import tensorflow as tf


def run():
    v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
    print("\nv.assign(2*v)\n", v.assign(2 * v))  # assigns new value to the variable
    print("\nv[0, 1].assign(42)\n", v[0, 1].assign(42))  # assign to specific index
    print("\nv[:, 2].assign([0., 1.])\n", v[:, 2].assign([0., 1.]))  # assign to range

    try:
        v[1] = [7., 8., 9.]  # can't assign without assign()
    except TypeError as ex:
        print("\nv[1] = [7., 8., 9.]\n", ex)

    # scatter_nd_update() applies sparse updates to individual values or slices in a tf.Variable
    # adds values specified by 'updates' into locations specified by 'indices'
    print("\nv.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100., 200.])\n",
          v.scatter_nd_update(indices=[[0, 0], [1, 2]],
                              updates=[100., 200.]))

    # creates a slice of values that can be applied to the variable using scatter_update()
    sparse_delta = tf.IndexedSlices(values=[[1., 2., 3.], [4., 5., 6.]],
                                    indices=[1, 0])
    print(v.scatter_update(sparse_delta))

    print("\n end variables")
