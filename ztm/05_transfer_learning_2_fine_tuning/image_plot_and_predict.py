import os
import pathlib
import random

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt, image as mpimg

import data_utils as du
from transfer_learning_experiments import experiment_two


def get_random_image(dir, class_names):
    target_class = random.choice(class_names)
    target_dir = os.path.join(dir, target_class)

    random_image = random.choice(os.listdir(target_dir))
    random_image_path = os.path.join(target_dir, random_image)
    # img = mpimg.imread(random_image_path)
    img = tf.io.read_file(random_image_path)

    return img


def visualize_image(image, class_name):
    plt.imshow(image)
    plt.title(class_name)
    plt.axis("off")
    plt.show()


def predict_random(model, test_data):
    img = get_random_image(du.TEST_DATA_PATH, test_data.class_names)
    # img_pred = img / 255.
    img_pred = tf.image.decode_image(img)
    img_pred = tf.image.resize(img_pred, [du.IMG_SIZE, du.IMG_SIZE])
    img_pred = img_pred / 255.

    pred = model.predict(tf.expand_dims(img_pred, axis=0))
    class_name = test_data.class_names[np.argmax(pred)]
    print(pred)

    visualize_image(img_pred, class_name)


def run():
    train_data = tf.keras.preprocessing.image_dataset_from_directory(du.TRAIN_DATA_PATH,
                                                                     label_mode="categorical",
                                                                     image_size=du.IMG_SHAPE)
    test_data = tf.keras.preprocessing.image_dataset_from_directory(du.TEST_DATA_PATH,
                                                                    label_mode="categorical",
                                                                    image_size=du.IMG_SHAPE)
    print(train_data.class_names)


    model, history = experiment_two(train_data, test_data)

    # img = get_random_image(du.TEST_DATA_PATH, test_data.class_names)
    # img_pred = img / 255.
    # img_pred = tf.image.decode_image(img)
    # img_pred = tf.image.resize(img_pred, [du.IMG_SIZE, du.IMG_SIZE])
    # img_pred = img_pred / 255.

    # pred = model.predict(tf.expand_dims(img_pred, axis=0))
    # class_name = test_data.class_names[np.argmax(pred)]
    # print(pred)

    # visualize_image(img_pred, class_name)
    # print(model.evaluate(test_data))

    predict_random(model, test_data)
    predict_random(model, test_data)
    predict_random(model, test_data)
    predict_random(model, test_data)
    predict_random(model, test_data)
    predict_random(model, test_data)



