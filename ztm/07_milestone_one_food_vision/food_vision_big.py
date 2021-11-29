"""
TensorFlow Datasets:
    ready-to-use datasets
    Already in tensor format
    Well-established datasets for many problem types
    Consistent
However, TF Datasets are static, they do not change like real-world datasets would
"""

import tensorflow as tf
import matplotlib as plt
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
        print(image)


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

