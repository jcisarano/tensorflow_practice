# pizza vs steak classification
import os
import pathlib
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import fetch_data as fd

import tensorflow as tf


# images are from food 101 dataset, but reduced to include only images of pizza and steak for now
# Starting with a smaller dataset allows quicker experimentation
DATA_PATH: str = "https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip"
LOCAL_SAVE_PATH: str = os.path.join("datasets", "images")
LOCAL_FILE_NAME: str = "steak_pizza.zip"


def get_class_names():
    path = os.path.join(LOCAL_SAVE_PATH, "pizza_steak/train")
    data_dir = pathlib.Path(path)
    class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
    print(class_names)


def view_random_image(target_dir, target_class, show=True):
    target_folder = os.path.join(target_dir, target_class)
    random_image = random.sample(os.listdir(target_folder), 1)
    img = mpimg.imread(os.path.join(target_folder, random_image[0]))
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    print(f"Image shape: {img.shape}")
    if show:
        plt.show()

    return img


def run():
    # fd.fetch_remote_data(DATA_PATH, LOCAL_SAVE_PATH, LOCAL_FILE_NAME)
    fd.examine_files(os.path.join(LOCAL_SAVE_PATH, "pizza_steak"))
    get_class_names()

    img = view_random_image(os.path.join(LOCAL_SAVE_PATH, "pizza_steak/train"), "pizza")
    t_img = tf.constant(img)
    print(t_img[0]/255)
    print(t_img[0])

