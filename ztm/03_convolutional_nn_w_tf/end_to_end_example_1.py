import os

import matplotlib.pyplot as plt
import food_vision as fv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv2D, Dense, MaxPool2D, Activation
from tensorflow.keras.optimizers import Adam

import pandas as pd

LOCAL_SAVE_PATH: str = os.path.join("datasets", "images")
TRAIN_DATA_PATH: str = os.path.join(LOCAL_SAVE_PATH, "pizza_steak/train")
TEST_DATA_PATH: str = os.path.join(LOCAL_SAVE_PATH, "pizza_steak/test")

IMG_SIZE: tuple = (224, 224)


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


def load_minibatch_data(train_dir=TRAIN_DATA_PATH, test_dir=TEST_DATA_PATH, img_size=IMG_SIZE):
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
                                                   target_size=img_size,
                                                   class_mode="binary",
                                                   batch_size=32)
    test_data = test_datagen.flow_from_directory(directory=test_dir,
                                                 target_size=img_size,
                                                 class_mode="binary",
                                                 batch_size=32)
    return train_data, test_data


def load_minibatch_data_augmented(train_dir=TRAIN_DATA_PATH, test_dir=TEST_DATA_PATH, img_size=IMG_SIZE):
    train_datagen_augmented = ImageDataGenerator(rescale=1 / 255.,
                                                 rotation_range=0.2,
                                                 shear_range=0.2,
                                                 zoom_range=0.2,
                                                 width_shift_range=0.2,
                                                 height_shift_range=0.3,
                                                 horizontal_flip=True)

    train_datagen = ImageDataGenerator(rescale=1 / 255.)
    test_datagen = ImageDataGenerator(rescale=1 / 255.)

    train_data_augmented = train_datagen_augmented.flow_from_directory(directory=train_dir,
                                                                       target_size=img_size,
                                                                       class_mode="binary",
                                                                       batch_size=32)

    train_data = train_datagen.flow_from_directory(directory=train_dir,
                                                   target_size=img_size,
                                                   class_mode="binary",
                                                   batch_size=32)
    test_data = test_datagen.flow_from_directory(directory=test_dir,
                                                 target_size=img_size,
                                                 class_mode="binary",
                                                 batch_size=32)

    return train_data, train_data_augmented, test_data


def create_and_compile_baseline_model():
    model = Sequential([
        Conv2D(filters=10,
               # how many filters will pass over the input tensor (e.g. sliding windows on an image), higher = more complex model
               kernel_size=3,
               # determines shape of the filter (sliding window) over the output, lower vals learn smaller features,
               strides=1,  # number of steps the filter will take at a time
               padding="valid",  # "same" will pad tensor so that output shape is same as input, "valid" will not pad
               activation="relu",
               input_shape=(224, 224, 3)),  # input layer (specify input shape)
        Conv2D(10, 3, activation="relu"),
        Conv2D(10, 3, activation="relu"),
        Flatten(),
        Dense(1, activation="sigmoid")  # output layer, binary classification, so only 1 output neuron
    ])
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(),
                  metrics=["accuracy"])

    return model


def create_and_compile_better_baseline_model():
    model = Sequential([
        Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
        MaxPool2D(pool_size=2),
        # max pooling keeps highest val in small grid, e.g. 4x4, reduces size and keeps most important
        Conv2D(10, 3, activation="relu", ),
        MaxPool2D(),
        Conv2D(10, 3, activation="relu", ),
        MaxPool2D(),
        Flatten(),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    return model


def fit_model(model, train_data, val_data):
    return model.fit(train_data,
                     epochs=5,
                     steps_per_epoch=len(train_data),
                     validation_data=val_data,
                     validation_steps=len(val_data))


def plot_training_curve(history):
    pd.DataFrame(history.history).plot(figsize=(10, 7))
    plt.show()


def plot_loss_curve(history):
    """
    Returns separate loss curves for training and validation metrics
    :param history:
    :return:
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(len(history.history["loss"]))

    # plot loss
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()

    plt.figure()
    # plot accuracy
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()


def run():
    # visualize_random_image()
    train_data, test_data = load_minibatch_data()

    # create a simple model as comparison/baseline for future experimentation
    # baseline_model = create_and_compile_baseline_model()
    # print(baseline_model.summary())

    # baseline_history = fit_model(model=baseline_model, train_data=train_data, val_data=test_data)

    # plot_training_curve(baseline_history)
    # plot_loss_curve(baseline_history)
    # baseline model seems to be overfitting the data: the training loss decreases, while the validation loss increases
    # loss curves for training and validation should have similar shapes showing improvement

    # ways to cause overfitting:
    # increase number of conv layers
    # increase number of filters per layer
    # add another dense layer to output of flattened layer

    # ways to reduce overfitting:
    # add data augmentation
    # add regularization layers (e.g. MaxPool2D)
    # train more data

    # new baseline to reduce overfitting
    baseline_model_1 = create_and_compile_better_baseline_model()
    print(baseline_model_1.summary())
    baseline_history_1 = fit_model(model=baseline_model_1, train_data=train_data, val_data=test_data)
    plot_loss_curve(baseline_history_1)
    # max pooling not only improves accuracy, it it reduces overfitting: the curves look better

    # DATA AUGMENTATION
    train_data, train_data_augmented, test_data = load_minibatch_data_augmented()
