"""
Dataset is Kaggle intro to NLP dataset: tweets labeled as disaster/not disaster. Original source available
at Kaggle: https://www.kaggle.com/c/nlp-getting-started
"""

from helper_functions import create_tensorboard_callback, plot_loss_curves, compare_histories

import pandas as pd
import random

TRAIN_PATH: str = "datasets/train.csv"
TEST_PATH: str = "datasets/test.csv"


def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    # print(train_df.head)
    # print(train_df["text"][1])

    # dataset is relatively balanced between classes
    # if it were not, see https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    # print(train_df.target.value_counts())

    train_df_shuffled = train_df.sample(frac=1, random_state=42)

    # visualize random training examples
    rand_index = random.randint(0, len(train_df)-5)
    for row in train_df_shuffled[["text", "target"]][rand_index:rand_index+5].itertuples():
        _, text, target = row
        print(f"Target: {target}", "(disaster)" if target > 0 else "(not disaster)")
        print(f"Text: {text}")
        print("-----")

def run():
    print("nlp fundies")
    load_data()


