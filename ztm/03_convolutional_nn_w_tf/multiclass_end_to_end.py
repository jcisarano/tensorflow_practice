# Complete multiclass image classification system
# Train model to identify ten different classes of food.
# Base data will come from Food 101 dataset

"""
Steps in multiclass classification. They are similar to any ML problem.
    1. Explore the data. Become one with the data.
    2. Preprocess the data.
    3. Create the model. Start with a baseline to compare against.
    4. Evaluate the model.
    5. Adjust hyperparameters and improve the model, e.g. to beat the baseline and reduce overfitting.
    6. Repeat.
"""
import os

LOCAL_SAVE_PATH: str = os.path.join("datasets", "images")
LOCAL_DATA_PATH: str = os.path.join("datasets", "images/10_food_classes_all_data")
TRAIN_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "train")
TEST_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "test")


def walk_the_data():
    for dirpath, dirnames, filenames in os.walk(LOCAL_DATA_PATH):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")


def run():
    walk_the_data()
