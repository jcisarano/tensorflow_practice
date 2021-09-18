import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer

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

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])
housing_prepared = full_pipeline.fit_transform(housing)  # remember housing is training set w/o labels

svr_reg = SVR()
svr_reg.fit(housing_prepared, housing_labels)  # trains the regressor model
housing_predictions = svr_reg.predict(housing_prepared)
svr_mse = mean_squared_error(housing_labels, housing_predictions)
svr_rmse = np.sqrt(svr_mse)
print("SVR Regressor:", svr_rmse)