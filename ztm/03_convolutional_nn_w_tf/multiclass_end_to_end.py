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


def run():
    # Step 1: Visualize the data
    walk_the_data()
    class_names = get_class_names(TRAIN_DATA_PATH)
    print(class_names)
    img = food_vision.view_random_image(target_dir=TRAIN_DATA_PATH,
                                        target_class=random.choice(class_names))

    # Step 2: Preprocess the data
    train_data, test_data = load_minibatch_data(train_dir=TRAIN_DATA_PATH, test_dir=TEST_DATA_PATH)

    # Step 3: Create the baseline CNN model
    model = baseline_model()

    # Step 4: Fit the model
    baseline_history = model.fit(train_data,
                                 epochs=5,
                                 steps_per_epoch=len(train_data),
                                 validation_data=test_data,
                                 validation_steps=len(test_data),
                                 workers=-1, use_multiprocessing=True)

    # 5. Evaluate the model