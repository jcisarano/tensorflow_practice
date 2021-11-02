import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime

LOCAL_SAVE_PATH: str = os.path.join("datasets", "images")
LOCAL_DATA_PATH: str = os.path.join("datasets", "images/10_food_classes_10_percent")
TRAIN_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "train")
TEST_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "test")

IMG_SIZE: int = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE: int = 32


def list_filecount_in_dir(dir):
    for dirpath, dirnames, filenames in os.walk(dir):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}.")


def load_and_prep_data(train_dir=TRAIN_DATA_PATH, test_dir=TEST_DATA_PATH, batch_size=BATCH_SIZE):
    train_datagen = ImageDataGenerator(rescale=1/255.)
    test_datagen = ImageDataGenerator(rescale=1/255.)

    print("Train data:")
    train_data = train_datagen.flow_from_directory(train_dir,
                                                   target_size=IMG_SHAPE,
                                                   batch_size=batch_size,
                                                   class_mode="categorical")
    print("Test data:")
    test_data = test_datagen.flow_from_directory(test_dir,
                                                 target_size=IMG_SHAPE,
                                                 batch_size=batch_size,
                                                 class_mode="categorical")

    return train_data, test_data

# Callbacks
# callbacks are extra functionality you can add to a model to be performed during or after training. Some examples:
# Tracking experiments with TensorBoard callback
# Save weights, etc along the way with ModelCheckpoint callback
# Stopping a model training too long and overfits using EarlyStopping callback
# see tf.keras.callbacks.Callback


def create_tensorboard_callback(save_dir, experiment_name):
    log_dir = os.path.join(save_dir, experiment_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Will save TensorBoard log files to {log_dir}")
    return tensorboard_callback

