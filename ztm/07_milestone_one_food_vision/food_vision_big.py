"""
TensorFlow Datasets:
    ready-to-use datasets
    Already in tensor format
    Well-established datasets for many problem types
    Consistent
However, TF Datasets are static, they do not change like real-world datasets would
"""

import tensorflow_datasets as tfds

from helper_functions import compare_histories


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
    # print(train_data.shape)