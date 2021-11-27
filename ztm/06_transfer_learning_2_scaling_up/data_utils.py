import os

LOCAL_SAVE_PATH: str = os.path.join("datasets", "images")
LOCAL_DATA_PATH_10_PERCENT: str = os.path.join("datasets", "images/101_food_classes_10_percent")
TRAIN_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH_10_PERCENT, "train")
TEST_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH_10_PERCENT, "test")

DATA_PATH_CUSTOM: str = os.path.join(LOCAL_SAVE_PATH, "custom_food_images")

IMG_SIZE: int = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE: int = 32
