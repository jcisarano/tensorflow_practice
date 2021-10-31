# Complete multiclass image classification system
# Train model to identify ten different classes of food.
# Base data will come from Food 101 dataset

"""
Steps in multiclass classification. They are similar to any ML problem.
    1. Explore the data. Become one with the data.
    2. Preprocess the data.
    3. Create the model. Start with a baseline to compare against.
    4. Evaluate the model.
    5. Adjust hyperparameters and improve the model, e.g. to beat the baseline and reduce overfitting.
    6. Repeat.
"""
import os
import random

import numpy as np
import pathlib

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd

import food_vision

LOCAL_SAVE_PATH: str = os.path.join("datasets", "images")
LOCAL_DATA_PATH: str = os.path.join("datasets", "images/10_food_classes_all_data")
TRAIN_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "train")
TEST_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "test")

IMG_SIZE: int = 224


def walk_the_data():
    for dirpath, dirnames, filenames in os.walk(LOCAL_DATA_PATH):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")


def get_class_names(directory):
    data_dir = pathlib.Path(directory)
    # class_names = np.array([item.name for item in data_dir.glob('*')])
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))

    return class_names


def load_minibatch_data(train_dir=TRAIN_DATA_PATH, test_dir=TEST_DATA_PATH, do_shuffle=True, img_size=IMG_SIZE):
    train_datagen = ImageDataGenerator(rescale=1 / 255.)
    test_datagen = ImageDataGenerator(rescale=1 / 255.)

    test_data = test_datagen.flow_from_directory(directory=test_dir,
                                                 target_size=(img_size, img_size),
                                                 class_mode="categorical",
                                                 batch_size=32,
                                                 shuffle=do_shuffle
                                                 )
    train_data = train_datagen.flow_from_directory(directory=train_dir,
                                                   target_size=(img_size, img_size),
                                                   class_mode="categorical",
                                                   batch_size=32,
                                                   shuffle=do_shuffle
                                                   )
    return train_data, test_data


def load_minibatch_data_augmented(train_dir=TRAIN_DATA_PATH, test_dir=TEST_DATA_PATH, do_shuffle=True, img_size=IMG_SIZE):
    train_datagen = ImageDataGenerator(rescale=1/255.,
                                       rotation_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1/255.)

    train_data = train_datagen.flow_from_directory(directory=train_dir,
                                                   target_size=(img_size,img_size),
                                                   class_mode="categorical",
                                                   batch_size=32,
                                                   shuffle=do_shuffle)
    test_data = test_datagen.flow_from_directory(directory=test_dir,
                                                 target_size=(img_size,img_size),
                                                 class_mode="categorical",
                                                 batch_size=32,
                                                 shuffle=do_shuffle)
    return train_data, test_data


# baseline model matches the one on CNN Explorer site
def baseline_model(shape=(IMG_SIZE, IMG_SIZE, 3)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=10, kernel_size=3, input_shape=shape),
        tf.keras.layers.Activation(activation="relu"),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax")  # 10 because there are 10 categories
    ])

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    return model


def simplified_model(shape=(IMG_SIZE,IMG_SIZE,3)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(10, 3, input_shape=shape),
        tf.keras.layers.Activation(activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    return model


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
    _, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.sca(axes[0])
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()

    # plot accuracy

    xmax = max(val_loss + accuracy)
    xmin = min(val_loss + accuracy)
    ymin = 0
    ymax = len(val_loss)
    plt.sca(axes[1])
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.axis([xmin, xmax, ymin, ymax])
    plt.legend()
    plt.show()


def load_and_preprocess_img(path, img_shape=224):
    # load the image
    img = tf.io.read_file(path)
    # decode the image into a tensor
    img = tf.image.decode_image(img)
    # resize
    img = tf.image.resize(img, [img_shape, img_shape])
    # normalize image values
    img = img / 255.

    return img


def pred_and_plot_multiclass(model, filename, class_names):
    """
    :param model:
    :param filename:
    :param class_names:
    :return:
    """
    img = load_and_preprocess_img(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))
    pred_class = class_names[tf.argmax(pred[0])]

    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)
    plt.show()


def run():
    # Step 1: Visualize the data
    walk_the_data()
    class_names = get_class_names(TRAIN_DATA_PATH)
    print(class_names)
    # img = food_vision.view_random_image(target_dir=TRAIN_DATA_PATH,
    #                                     target_class=random.choice(class_names))

    # Step 2: Preprocess the data
    # train_data, test_data = load_minibatch_data(train_dir=TRAIN_DATA_PATH, test_dir=TEST_DATA_PATH)
    train_data, test_data = load_minibatch_data_augmented(train_dir=TRAIN_DATA_PATH, test_dir=TEST_DATA_PATH)

    # Step 3: Create the baseline CNN model
    model = baseline_model()
    # model = simplified_model()

    # Step 4: Fit the model
    baseline_history = model.fit(train_data,
                                 epochs=5,
                                 steps_per_epoch=len(train_data),
                                 validation_data=test_data,
                                 validation_steps=len(test_data),
                                 workers=-1, use_multiprocessing=True)

    # 5. Evaluate the model
    print(model.evaluate(test_data))

    plot_loss_curve(baseline_history)

    # 6. Adjust model hyperparameters to beat the baseline and reduce overfitting
    """Some methods to prevent overfitting:
        1. Train more data
        2. Simplify model, e.g. remove some layers or reduce num of hidden units
        3. Use data augmentation
        4. Use transfer learning (uses training from another, similar model/data on your dataset)
    """

    # 7. Continue to experiment to improve performance
    """
        1. Change model architecture (layers, hidden units)
        2. Adjust learning rate
        3. Change data augmentation hyperparams
        4. Train for longer
        5. Transfer learning (covered in future section)
    """

    # Test against other images
    img_0 = os.path.join(LOCAL_DATA_PATH, "03-hamburger.jpeg")
    img_1 = os.path.join(LOCAL_DATA_PATH, "03-pizza-dad.jpeg")
    img_2 = os.path.join(LOCAL_DATA_PATH, "03-steak.jpeg")
    img_3 = os.path.join(LOCAL_DATA_PATH, "03-sushi.jpeg")

    pred_and_plot_multiclass(model, img_0, class_names)
    pred_and_plot_multiclass(model, img_1, class_names)
    pred_and_plot_multiclass(model, img_2, class_names)
    pred_and_plot_multiclass(model, img_3, class_names)




