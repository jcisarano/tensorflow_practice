import os

import pandas as pd


def load_data(path, filename):
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path)


"""
json example?
https://stackoverflow.com/questions/38381887/how-to-read-json-files-in-tensorflow
"""