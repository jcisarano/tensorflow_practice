import os

LOCAL_SAVE_PATH: str = os.path.join("datasets", "images")
LOCAL_DATA_PATH: str = os.path.join("datasets", "images/10_food_classes_10_percent")
TRAIN_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "train")
TEST_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "test")

IMG_SIZE: int = 224


def list_filecount_in_dir(dir):
    for dirpath, dirnames, filenames in os.walk(dir):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}.")
