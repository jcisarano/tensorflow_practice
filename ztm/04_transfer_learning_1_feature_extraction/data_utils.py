import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

    train_data = train_datagen.flow_from_directory(train_dir,
                                                   target_size=IMG_SHAPE,
                                                   batch_size=batch_size,
                                                   class_mode="categorical")
    test_data = test_datagen.flow_from_directory(test_dir,
                                                 target_size=IMG_SHAPE,
                                                 batch_size=batch_size,
                                                 class_mode="categorical")

    return train_data, test_data

