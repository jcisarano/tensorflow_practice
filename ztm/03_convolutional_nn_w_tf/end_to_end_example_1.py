import os

import matplotlib.pyplot as plt
import food_vision as fv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv2D, Dense, MaxPool2D, Activation
from tensorflow.keras.optimizers import Adam



LOCAL_SAVE_PATH: str = os.path.join("datasets", "images")
TRAIN_DATA_PATH: str = os.path.join(LOCAL_SAVE_PATH, "pizza_steak/train")
TEST_DATA_PATH: str = os.path.join(LOCAL_SAVE_PATH, "pizza_steak/test")


def visualize_random_image():
    """
    look at random images to help understand what the training set contains
    :return:
    """
    plt.figure()
    plt.subplot(1, 2, 1)
    steak_img = fv.view_random_image(TRAIN_DATA_PATH, "steak", show=False)
    plt.subplot(1, 2, 2)
    pizza_img = fv.view_random_image(TRAIN_DATA_PATH, "pizza")


def load_minibatch_data(train_dir=TRAIN_DATA_PATH, test_dir=TEST_DATA_PATH):
    """
    load data, regularize it, and split into mini batches
    split data into batches to make sure it fits into memory
    splitting helps improve training, large sets may not train well
    :return:
    """
    # regularize data and split into batches
    # ImageDataGenerator can do realtime data augmentation also
    train_datagen = ImageDataGenerator(rescale=1 / 255.)
    test_datagen = ImageDataGenerator(rescale=1 / 255.)

    train_data = train_datagen.flow_from_directory(directory=train_dir,
                                                   target_size=(224, 224),
                                                   class_mode="binary",
                                                   batch_size=32)
    test_data = test_datagen.flow_from_directory(directory=test_dir,
                                                 target_size=(224, 224),
                                                 class_mode="binary",
                                                 batch_size=32)
    return train_data, test_data


def create_and_compile_baseline_model():
    model = Sequential([
        Conv2D(filters=10,
               kernel_size=3,
               strides=1,
               padding="valid",
               activation="relu",
               input_shape=(244, 244, 3)),  # input layer (specify input shape)
        Conv2D(10, 3, activation="relu"),
        Conv2D(10, 3, activation="relu"),
        Flatten(),
        Dense(1, activation="sigmoid")  # output layer, binary classification, so only 1 output neuron
    ])
    model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    return model


def run():
    # visualize_random_image()
    train_data, test_data = load_minibatch_data()

    # create a simple model as comparison/baseline for future experimentation
    baseline_model = create_and_compile_baseline_model()
