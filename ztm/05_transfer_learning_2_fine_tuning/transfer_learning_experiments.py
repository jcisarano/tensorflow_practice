
import  tensorflow as tf

import data_utils as du
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir


def run():
    # walk_through_dir(du.LOCAL_DATA_PATH_1_PERCENT)

    train_data_1_percent = tf.keras.preprocessing.image_dataset_from_directory(du.TRAIN_DATA_PATH_1_PERCENT,
                                                                               label_mode="categorical",
                                                                               image_size=du.IMG_SHAPE,
                                                                               batch_size=du.BATCH_SIZE)

    test_data = tf.keras.preprocessing.image_dataset_from_directory(du.TEST_DATA_PATH_1_PERCENT,
                                                                    label_mode="categorical",
                                                                    image_size=du.IMG_SHAPE,
                                                                    batch_size=du.BATCH_SIZE)

