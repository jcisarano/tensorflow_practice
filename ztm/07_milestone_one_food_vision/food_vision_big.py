"""
TensorFlow Datasets:
    ready-to-use datasets
    Already in tensor format
    Well-established datasets for many problem types
    Consistent
However, TF Datasets are static, they do not change like real-world datasets would
"""

"""
Neural networks perform best when data is formatted properly, e.g. batched, normalized, etc
However, raw data is rarely formatted as you will need, so you will need to write preprocessing functions.

What we know about our food data:
    unit8 datatype
    images are different sizes
    not normalized-- the color values range from 0 to 255
    
However, the models like data in the format:
    data in float32 type
    all tensors should be the same size
    tensors with normalized color values (0-1) usually work better
    
We'll use EfficientNetBX pretrained model, so we won't need to rescale data--it has scaling layer built in.
Our function will need to:
    1) reshape images
    2) change datatype from unit8 to float32
"""

"""
*** Very important resource on data import pipeline in TensorFlow: https://www.tensorflow.org/guide/data ***
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from helper_functions import compare_histories


def visualize_data(train_data, test_data, ds_info):
    """
    Explore the data to find:
        Class names
        Shape of input data (image tensors)
        Data type of input data
        What do the labels look like? One-hot encoding or label encoding?
        Do the labels match the class names?
    """
    print(ds_info.features["label"].names[:10])
    train_one_sample = train_data.take(1)
    plt.figure(figsize=(8, 5))
    for image, label in train_one_sample:
        print(f"""
        Image shape: {image.shape}
        Image datatype: {image.dtype}
        Target class from Food101 (tensor form): {label}
        Class name (str form): {ds_info.features["label"].names[label.numpy()]}
        """)
        # print(image)
        plt.subplot(121)
        plt.imshow(image)
        plt.title(f"class: {ds_info.features['label'].names[label.numpy()]}")
        plt.axis(False)
        print(f"Image before preprocessing:\n{image[:2]}...,\nShape: {image.shape},\nDataType: {image.dtype}\n")
        preprocessed_img = preprocess_img(image, label)
        print(f"Image after preprocessing:\n{preprocessed_img[:2]}...,\nShape: {preprocessed_img.shape},\n"
              f"DataType: {preprocessed_img.dtype}\n")
        plt.subplot(122)
        plt.imshow(preprocessed_img / 255.)
        plt.title(f"Preprocessed version")
        plt.axis(False)
        plt.show()


def preprocess_img(image, label, target_img_shape=224):
    """
    Converts img datatype to float32 and reshapes image to target_img_shape x target_img_shape X num_color_channels
    :param image:
    :param label:
    :param target_img_shape:
    :return:
    """
    image = tf.image.resize(image, [target_img_shape, target_img_shape])
    return tf.cast(image, tf.float32), label


def run():
    # a new way to load food101 dataset, from tensorflow_datasets
    datasets_list = tfds.list_builders()
    print("food101" in datasets_list)

    # load the full dataset, but it will take some time because the dataset is 5-6 gb
    # other datasets can be much larger, so check the dataset size first
    (train_data, test_data), ds_info = tfds.load(name="food101",
                                                 split=["train", "validation"],
                                                 shuffle_files=True,
                                                 as_supervised=True,  # includes labels in tuple (data,labels)
                                                 with_info=True  # includes meta data
                                                 )

    # visualize_data(train_data, test_data, ds_info)

    # map preprocessing function to training data (and parallelize)
    train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    # shuffle train_data and turn it into batches and prefetch it (to load faster)
    # 1000 is good value, but larger vals can be limited by available RAM
    train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Map preprocessing function also for test_data
    test_data = test_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

    print(train_data, test_data)
