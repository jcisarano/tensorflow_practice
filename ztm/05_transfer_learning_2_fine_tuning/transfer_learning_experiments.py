
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random

import data_utils as du
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir


def run():
    # walk_through_dir(du.LOCAL_DATA_PATH_1_PERCENT)

    # load datasets from files
    train_data_1_percent = tf.keras.preprocessing.image_dataset_from_directory(du.TRAIN_DATA_PATH_1_PERCENT,
                                                                               label_mode="categorical",
                                                                               image_size=du.IMG_SHAPE,
                                                                               batch_size=du.BATCH_SIZE)

    test_data = tf.keras.preprocessing.image_dataset_from_directory(du.TEST_DATA_PATH_1_PERCENT,
                                                                    label_mode="categorical",
                                                                    image_size=du.IMG_SHAPE,
                                                                    batch_size=du.BATCH_SIZE)

    # do data augmentation
    # tf.keras.layers.experimental.preprocessing has data augmentation features
    # Using Sequential() here to pass to functional API
    # This will add data augmentation to model, so they run on the gpu and become part of the model,
    # so a saved model will include the preprocessing steps
    # this preprocessing is only used during training, not during evaluation
    data_augmentation = keras.Sequential([
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.2),
        preprocessing.RandomZoom(0.2),
        preprocessing.RandomHeight(0.2),
        preprocessing.RandomWidth(0.2),
        # preprocessing.Rescaling(1./255)  # use for models like ResNet50V2, but not EfficientNet
    ], name="data_augmentation")

    # visualize data augmentation
    # view a random image and compare to augmented version
    target_class = random.choice(train_data_1_percent.class_names)
    target_dir = os.path.join(du.TRAIN_DATA_PATH_1_PERCENT, target_class)

    random_image = random.choice(os.listdir(target_dir))
    random_image_path = os.path.join(target_dir, random_image)
    img = mpimg.imread(random_image_path)
    plt.imshow(img)
    plt.show()


