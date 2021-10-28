import os

import matplotlib.pyplot as plt
import food_vision as fv

LOCAL_SAVE_PATH: str = os.path.join("datasets", "images")


def visualize_random():
    plt.figure()
    plt.subplot(1, 2, 1)
    steak_img = fv.view_random_image(os.path.join(LOCAL_SAVE_PATH, "pizza_steak/train"), "steak", show=False)
    plt.subplot(1, 2, 2)
    pizza_img = fv.view_random_image(os.path.join(LOCAL_SAVE_PATH, "pizza_steak/train"), "pizza")


def run():
    visualize_random()
