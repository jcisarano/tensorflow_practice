import tensorflow as tf
from tensorflow_datasets.video.bair_robot_pushing import IMG_SHAPE

from transfer_learning import load_and_prep_data, TRAIN_DATA_PATH, TEST_DATA_PATH


# import tensorflow_datasets as tfds


# def tfds_load_example():
#     datasets_list = tfds.list_builders()


def run():
    # loads in batches using ImageGenerator
    # train_data, test_data = load_and_prep_data()

    # more current version, loads image data without need to create ImageGenerator
    train_data = tf.keras.preprocessing.image_dataset_from_directory(TRAIN_DATA_PATH,
                                                                     label_mode="categorical",
                                                                     image_size=(224, 224),
                                                                     batch_size=32)
    test_data = tf.keras.preprocessing.image_dataset_from_directory(TEST_DATA_PATH,
                                                                    label_mode="categorical",
                                                                    image_size=(224,224),
                                                                    batch_size=32)

    print(train_data)
    print(test_data)

    print(train_data.class_names)

    print("batch load")
