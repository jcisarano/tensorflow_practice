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
    for image, label in train_one_sample:
        print(f"""
        Image shape: {image.shape}
        Image datatype: {image.dtype}
        Target class from Food101 (tensor form): {label}
        Class name (str form): {ds_info.features["label"].names[label.numpy()]}
        """)
        # print(image)
        plt.figure()
        plt.imshow(image)
        plt.title(f"class: {ds_info.features['label'].names[label.numpy()]}")
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
    return tf.cast(image, tf.float32)


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

    visualize_data(train_data, test_data, ds_info)

    sample = train_data.take(1)
    for image, label in sample:
        print(f"Image before preprocessing:\n{image[:2]}...,\nShape: {image.shape},\nDataType: {image.dtype}\n")
        preprocessed_img = preprocess_img(image, label)
        print(f"Image after preprocessing:\n{preprocessed_img[:2]}...,\nShape: {preprocessed_img.shape},\nDataType: {preprocessed_img.dtype}\n")


