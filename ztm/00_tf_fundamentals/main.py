# This project covers fundamental concepts of tensors using TensorFlow
# Will cover:
#   Introduction to tensors
#   Getting info from tensors
#   Manipulating tensors
#   Tensors and NumPy
#   Using @tf.function (a way to speed up regular Python functions)
#   Using GPUs with TensorFlow

import tensorflow as tf

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(tf.__version__)

    # create tensors with tf.constant()
    scalar = tf.constant(7)
    print(scalar)

    # check the number of dimensions of a tensor
    # (ndim stands for number of dimensions)
    print(scalar.ndim)

    # create a vector
    vector = tf.constant([10, 10])
    print(vector)

    # dimensions of vector
    print(vector.ndim)

    # create a matrix - matrix has more than 1 dimension
    matrix = tf.constant([[10, 7],
                          [7, 10]])
    print(matrix)
    print(matrix.ndim)

    # create matrix while specifying datatype using dtype parameter
    another_matrix = tf.constant([[10., 7.],
                                  [3., 2.],
                                  [8., 9.]], dtype=tf.float16)

    print(another_matrix)
    print(another_matrix.ndim)

    # create a tensor
    tensor = tf.constant([[[1, 2, 3],
                           [4, 5, 6]],
                          [[7, 8, 9],
                           [10, 11, 12]],
                          [[13, 14, 15],
                           [16, 17, 18]]])
    print(tensor)
    print(tensor.ndim)

    # review
    # scalar is a single number
    # vector is a number with direction, e.g. wind speed and direction
    # matrix is 2-dimensional array of numbers
    # tensor is n-dimensional array of numbers, where n can be any number

    ### Creating tensors with tf.Variable
    changeable_tensor = tf.Variable([10, 7])
    unchangeable_tensor = tf.constant([10, 7])
    print(changeable_tensor, '\n', unchangeable_tensor)

    # use .assign() to change tensor created with .Variable():
    changeable_tensor[0].assign(7)
    print(changeable_tensor)
    # cannot do this to tensor created with .constant()

    # NOTE: tensorflow usually decides whether to make variable or constant. If in doubt, use constant and change later if needed.

    ### Create random tensors
    # random tensors can be any dimensions and are filled with random numbers
    # e.g. good for initializing random weights at start of learning
    random_1 = tf.random.Generator.from_seed(42)
    random_1 = random_1.normal(shape=(3, 2))  # normal distribution is symmetrical bell-shaped graph
    print(random_1)

    random_2 = tf.random.Generator.from_seed(42)
    random_2 = random_2.normal(shape=(3, 2))
    print(random_2)

    # seed means the two will be the same
    print(random_1 == random_2)

    ### Shuffle order of elements in a tensor
    # depending on the problem, it can be useful to have random distribution of training data
    not_shuffled = tf.constant([[10, 7],
                               [3, 4],
                               [1, 5], ]
                               )
    # print(not_shuffled.ndim)  # remember ndim is the number of dimensions in the shape
    print(not_shuffled)
    shuffled = tf.random.shuffle(not_shuffled) # randomly shuffles tensor along its first dimension
    print(shuffled)

    # tf.random.set_seed(42)
    shuffled = tf.random.shuffle(not_shuffled, seed=42)
    print(shuffled)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
