# food classifier using pretrained model from tf.keras.application
# https://www.tensorflow.org/api_docs/python/tf/keras/applications
import os
import tensorflow as tf

from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir

LOCAL_SAVE_PATH: str = os.path.join("datasets", "images")
LOCAL_DATA_PATH: str = os.path.join("datasets", "images/10_food_classes_10_percent")
TRAIN_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "train")
TEST_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "test")

IMG_SIZE: int = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE: int = 32


def run():
    # unzip_data("10_food_classes_10_percent.zip")
    # walk_through_dir(LOCAL_DATA_PATH)

    # returns BatchDataset, with batch size 32
    train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=TRAIN_DATA_PATH,
                                                                                label_mode="categorical",
                                                                                image_size=IMG_SHAPE,
                                                                                batch_size=BATCH_SIZE)
    test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=TEST_DATA_PATH,
                                                                    label_mode="categorical",
                                                                    image_size=IMG_SHAPE,
                                                                    batch_size=BATCH_SIZE)

    for images, labels in train_data_10_percent.take(1):
        print(images, labels)
