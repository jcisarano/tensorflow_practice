import os

import numpy as np
import pathlib

from keras_preprocessing.image import ImageDataGenerator

LOCAL_SAVE_PATH: str = os.path.join("datasets", "images")
# LOCAL_DATA_PATH: str = os.path.join("datasets", "images/10_food_classes_all_data")
LOCAL_DATA_PATH: str = os.path.join("datasets", "images/10_food_classes_10_percent")
TRAIN_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "train")
TEST_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "test")

IMG_SIZE: int = 224


def get_class_names(directory):
    data_dir = pathlib.Path(directory)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))

    return class_names


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
                                                   batch_size=64,
                                                   shuffle=do_shuffle)
    test_data = test_datagen.flow_from_directory(directory=test_dir,
                                                 target_size=(img_size,img_size),
                                                 class_mode="categorical",
                                                 batch_size=64,
                                                 shuffle=do_shuffle)
    return train_data, test_data


def run():
    # Step 1: Visualize the data
    class_names = get_class_names(TRAIN_DATA_PATH)
    print(class_names)
    # img = food_vision.view_random_image(target_dir=TRAIN_DATA_PATH,
    #                                     target_class=random.choice(class_names))

    # Step 2: Preprocess the data
    # train_data, test_data = load_minibatch_data(train_dir=TRAIN_DATA_PATH, test_dir=TEST_DATA_PATH)
    train_data, test_data = load_minibatch_data_augmented(train_dir=TRAIN_DATA_PATH, test_dir=TEST_DATA_PATH)
