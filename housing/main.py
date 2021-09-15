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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

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
    # not a great solution, since it will break if data order changes or more is added
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    # works, but depends on ids never changing, so new data must always be added to the end
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[-in_test_set], data[in_test_set]


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

    # creates new index column to use for train-test split
    raw_data_with_id = raw_data.reset_index()  # adds index column
    train_set, test_set = split_train_test_by_id(raw_data_with_id, 0.2, "index")
    print(len(train_set))
    print(len(test_set))

    # uses lat/long to generate index
    raw_data_with_id["id"] = raw_data["longitude"] * 1000 + raw_data["latitude"] * 1000
    train_set, test_set = split_train_test_by_id(raw_data_with_id, 0.2, "id")
    # consistent, but because lat/long are coarse, there is some overlap of ids:
    print(len(train_set))
    print(len(test_set))

    # sklearn has function that splits data randomly:
    train_set, test_set = train_test_split(raw_data, test_size=0.2, random_state=42)

    # creating train/test distribution based on income distribution
    raw_data["income_cat"] = pd.cut(raw_data["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])
    # raw_data["income_cat"].hist()
    # plt.show()

    # split based on distribution
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(raw_data, raw_data["income_cat"]):
        strat_train_set = raw_data.loc[train_index]
        strat_test_set = raw_data.loc[test_index]

    print(raw_data["income_cat"].value_counts() / len(raw_data))
    print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

    # no longer need income_cat column, so remove it from datasets
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    data_copy = strat_train_set.copy()
    data_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    plt.show()

    data_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                   s=data_copy["population"] / 100, label="population",
                   figsize=(10, 7), c="median_house_value", cmap=plt.get_cmap("jet"),
                   colorbar=True, )
    plt.legend()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
