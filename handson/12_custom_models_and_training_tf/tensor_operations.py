import tensorflow as tf


def run():
    # create tensor matrix
    print(tf.constant([[1., 2., 3.], [4., 5., 6.]]))
    print(tf.constant(42))
    t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
    print("\nt:\n", t)
    print("t.shape:", t.shape)
    print("t.dtype:", t.dtype)
    print("\nt[:, 1:]:\n", t[:, 1:])
    print("\nt[..., 1, tf.newaxis]:\n", t[..., 1, tf.newaxis])

    # + is equivalent to tf.add()
    print("t+10", t+10)
    print("tf.square(t):\n", tf.square(t))
    # @ is equivalent to tf.matmul()
    print("t @ tf.transpose(t):\n", t @ tf.transpose(t))

    from tensorflow import keras
    K = keras.backend
    print("\nK.square(K.transpose(t)) + 10\n", K.square(K.transpose(t)) + 10)

    import numpy as np
    a = np.array([2., 4., 5.])
    print("Numpy:\n", tf.constant(a))
    print("t.numpy():\n", t.numpy())
    print("np.array(t):\n", np.array(t))
    print("tf.square(a):\n", tf.square(a))
    print("np.square(t):\n", np.square(t))

    # type conflicts
    try:
        tf.constant(2.0) + tf.constant(40)
    except tf.errors.InvalidArgumentError as ex:
        print(ex)

    try:
        tf.constant(2.0) + tf.constant(40., dtype=tf.float64)
    except tf.errors.InvalidArgumentError as ex:
        print(ex)

    t2 = tf.constant(40., dtype=tf.float64)
    print(tf.constant(2.0) + tf.cast(t2, tf.float32))


    print("tensor operations")
