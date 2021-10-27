import os
import tarfile
import urllib.request
import zipfile


def fetch_remote_data(remote_data_url, local_save_path, local_filename):
    os.makedirs(local_save_path, exist_ok=True)
    full_local_path = os.path.join(local_save_path, local_filename)
    urllib.request.urlretrieve(remote_data_url, full_local_path)
    with zipfile.ZipFile(full_local_path, 'r') as zip_ref:
        zip_ref.extractall(local_save_path)
    # loaded_file = tarfile.open(full_local_path)
    # loaded_file.extractall(path=local_save_path)
    # loaded_file.close()


def load_data(path, filename):
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path)