# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32

DATA_SERVER_ROOT: str = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
LOCAL_SAVE_PATH: str = os.path.join("datasets", "housing")
REMOTE_DATA_URL: str = DATA_SERVER_ROOT + "datasets/housing/housing.tgz"
LOCAL_ZIP_FILENAME: str = "housing.tgz"
LOCAL_CSV_FILENAME: str = "housing.csv"


def fetch_remote_data(remote_data_url=REMOTE_DATA_URL, local_save_path=LOCAL_SAVE_PATH,
                      local_filename=LOCAL_ZIP_FILENAME):
    os.makedirs(local_save_path, exist_ok=True)
    full_local_path = os.path.join(local_save_path, local_filename)
    urllib.request.urlretrieve(remote_data_url, full_local_path)
    loaded_file = tarfile.open(full_local_path)
    loaded_file.extractall(path=local_save_path)
    loaded_file.close()


def load_data(path=LOCAL_SAVE_PATH, filename=LOCAL_CSV_FILENAME):
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    # not a great solution, since it is random and generates different sets each run
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fetch_remote_data()
    raw_data = load_data()
    # print(raw_data.head())
    # print(raw_data.info())
    # print(raw_data["ocean_proximity"].value_counts())
    # print(raw_data.describe())
    # raw_data.hist(bins=50, figsize=(20, 15))
    # plt.show()
    train_set, test_set = split_train_test(raw_data, 0.2)
    print(len(train_set))
    print(len(test_set))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
