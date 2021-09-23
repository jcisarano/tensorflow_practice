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
    shuffled = tf.random.shuffle(not_shuffled)  # randomly shuffles tensor along its first dimension
    print(shuffled)

    # tf.random.set_seed(42)
    shuffled = tf.random.shuffle(not_shuffled, seed=42)
    print(shuffled)

    ## Other ways to make tensors
    print(tf.ones([10, 7]))
    print(tf.zeros([10, 7]))

    # turn numpy arrays into tf tensors
    # tensors run much faster on gpu
    import numpy as np

    numpy_a = np.arange(1, 25, dtype=np.int32)
    print(numpy_a)

    # convert:
    A = tf.constant(numpy_a)
    print(A)

    # convert to 3 dimensional version
    # number of elements must match before and after, in this case 2*3*4=24 or 3*8=24
    A = tf.constant(numpy_a, shape=(2, 3, 4))
    print(A)

    # Tensor information
    # Shape - length or number of elements of each dimension, tf.shape
    # Rank - number of tensor dimensions, e.g. scalar is rank 0, vector rank 1, matrix rank 2, tensor rank n, tf.ndim
    # Axis or dimension - particular dimension of a tensor, tensor[0], tensor[:,1]
    # Size - total number of elements in the tensor, tf.size(tensor)

    # Create a rank 4 tensor (4 dimensions)
    rank_4_tensor = tf.zeros(shape=[2, 3, 4, 5])
    print(rank_4_tensor)
    print(rank_4_tensor[0])

    print('\n', rank_4_tensor.shape, rank_4_tensor.ndim, tf.size(rank_4_tensor))
    print("Datatype of every element:", rank_4_tensor.dtype)
    print("Number of dimensions (rank):", rank_4_tensor.ndim)
    print("Shape of tensor", rank_4_tensor.shape)
    print("Elements along the 0 axis:", rank_4_tensor.shape[0])
    print("Elements along the last axis:", rank_4_tensor.shape[-1])
    print("Total number of elements:", tf.size(rank_4_tensor))
    print("Total number of elements:", tf.size(rank_4_tensor).numpy())  # converts to number only

    # Indexing tensors works just like Python lists
    # Get the first two elements of each dimension:
    print(rank_4_tensor[:2, :2, :2, :2])

    # Get first element from each dimension from each index except get the whole last one:
    print(rank_4_tensor[:1, :1, :1, :])

    # same, but for the second to last:
    print(rank_4_tensor[:1, :1, :, :1])

    # same, but for the first:
    print(rank_4_tensor[:, :1, :1, :1], "\n\n\n")

    # create a rank 2 tensor:
    rank_2_tensor = tf.constant([[10, 7], [3, 4]])
    print(rank_2_tensor)
    print(rank_2_tensor.shape, rank_2_tensor.ndim, tf.size(rank_2_tensor))

    # get last element
    print(rank_2_tensor[:, -1])

    # add extra dimension to existing tensor w/o changing existing info:
    rank_3_tensor = rank_2_tensor[..., tf.newaxis]  # adds a new axis of 1 at the end, ... means all previous axes
    print(rank_3_tensor)

    print(tf.expand_dims(rank_2_tensor, axis=-1))  # same as above, adds last axis
    print(tf.expand_dims(rank_2_tensor, axis=0))  # expands 0 axis

    # Basic tensor operations
    # addition:
    tensor = tf.constant([[10, 7], [3, 4]])
    print(tensor + 10)  # original tensor is unchanged
    # multiplication
    print(tensor * 10)

    # tensorflow built-in arithmetic functions
    # tensorflow versions of most operators are faster on GPU, especially with big tensors
    print(tf.multiply(tensor, 10))
    print(tensor * tensor)  # element-wise multiplication

    # matrix multiplications - dot product
    print(tf.matmul(tensor, tensor))
    # same thing:
    print(tensor @ tensor)

    # tensors with different shapes
    X = tf.constant([[1, 2],
                     [3, 4],
                     [5, 6]])
    Y = tf.constant([[7, 8],
                     [9, 10],
                     [11, 12]])
    # sizes incompatible for dot product:
    # X @ Y
    # inner dimensions must match
    # resulting shape is outer dimensions

    # so change the shape of Y to make it work
    print(X @ tf.reshape(Y, shape=(2, 3)))

    # or
    print(tf.reshape(X, shape=(2, 3)) @ Y)

    # or, using transpose
    # however, transpose flips axes, so result is different from reshape()
    print(tf.transpose(X) @ Y)

    # dot product - another way to do it - same result as above
    print(tf.tensordot(tf.transpose(X), Y, axes=1))

    # each of these gives different results
    print(tf.matmul(X, tf.transpose(Y)))
    print(tf.matmul(X, tf.reshape(Y, shape=(2, 3))))

    print("\nNormal Y:", Y)
    print("Reshape Y (2,3):",
          tf.reshape(Y, shape=(2, 3)))  # reshape takes elements in order and fits them to the new shape
    print("Transpose Y:", tf.transpose(Y))  # transpose flips axes
    # generally, transpose is more useful, e.g. when trying to multiply two matrices that do not match shapes

    # tensor multiplication choices:
    # tf.multiply
    # tf.matmul
    # tf.tensordot
    # @

    # tensor data types
    B = tf.constant([1.7, 7.4])
    print(B)

    C = tf.constant([1, 7])
    print(C)

    # changing data type
    D = tf.cast(B, dtype=tf.float16)
    print(D, D.dtype)

    E = tf.cast(C, dtype=tf.float32)
    print(E, E.dtype)

    F = tf.cast(E, dtype=tf.float16)
    print(F, F.dtype)

    # absolute value
    D = tf.constant([-7, -10])
    print(tf.abs(D), "\n\n")

    # aggregating tensors
    # minimum of tensor, maximum of tensor, mean of tensor, sum of tensor

    E = tf.constant(np.random.randint(0, 100, size=50))
    print(tf.size(E), E.shape, E.ndim)
    print(tf.reduce_min(E))
    print(tf.reduce_max(E))
    print(tf.reduce_mean(E))
    print(tf.reduce_sum(E))

    print(tf.math.reduce_variance(tf.cast(E, dtype=tf.float32)))
    print(tf.math.reduce_std(tf.cast(E, dtype=tf.float32)), "\n\n")  # reduce_std requires float

    # positional max and min
    tf.random.set_seed(42)
    F = tf.random.uniform(shape=[50])
    print(F)
    print(tf.argmax(F))  # returns the index of the highest value
    print(F[tf.argmax(F)])
    print(tf.reduce_max(F))

    print(tf.argmin(F))  # returns the index of the lowest value
    print(F[tf.argmin(F)])
    print(tf.reduce_min(F), "\n\n")

    # squeezing a tensor
    # squeeze removes dimensions of size one
    tf.random.set_seed(42)
    G = tf.constant(tf.random.uniform(shape=[50]), shape=(1, 1, 1, 1, 50))
    print(G.shape)
    G_sq = tf.squeeze(G)
    print(G_sq.shape, "\n\n")

    # one hot encoding
    some_list = [0, 1, 2, 3, 1, 3, 0]  # represents list to encode
    print(tf.one_hot(some_list, depth=4))

    # specify custom values:
    print(tf.one_hot(some_list, depth=4, on_value="hello",off_value="goodbye"))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
