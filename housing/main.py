# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import tarfile
import urllib.request

DATA_SERVER_ROOT: str = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
LOCAL_SAVE_PATH: str = os.path.join("datasets", "housing")
REMOTE_DATA_URL: str = DATA_SERVER_ROOT + "datasets/housing/housing.tgz"
LOCAL_FILENAME: str = "housing.tgz"


def fetch_remote_data(remote_data_url=REMOTE_DATA_URL, local_save_path=LOCAL_SAVE_PATH, local_filename=LOCAL_FILENAME):
    os.makedirs(local_save_path, exist_ok=True)
    full_local_path = os.path.join(local_save_path, local_filename)
    urllib.request.urlretrieve(remote_data_url, full_local_path)
    loaded_file = tarfile.open(full_local_path)
    loaded_file.extractall(path=local_save_path)
    loaded_file.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fetch_remote_data()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
