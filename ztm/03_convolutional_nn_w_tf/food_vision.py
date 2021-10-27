# pizza vs steak classification
import os
import fetch_data as fd

DATA_PATH : str = "https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip"
LOCAL_SAVE_PATH : str = os.path.join("datasets", "images")
LOCAL_FILE_NAME: str = "steak_pizza.zip"


def run():
    # fd.fetch_remote_data(DATA_PATH, LOCAL_SAVE_PATH, LOCAL_FILE_NAME)
