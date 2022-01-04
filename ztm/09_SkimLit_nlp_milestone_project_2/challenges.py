"""
1. Turn test data samples into tf.data Dataset for fast loading and evaluate/predict best model on test samples
2. Find most wrong predictions from test dataset
3. Make example predictions on randomized control trial abstracts from the wild, find them on PubMed.gov, e.g. search
    there for "nutrition rct" or similar. There are also a few examples in extras directory of course github

Note: Pretrained model stored in Google drive for course if desired:
    https://storage.googleapis.com/ztm_tf_course/skimlit/skimlit_tribrid_model.zip
"""

import tensorflow as tf

MODEL_PATH: str = "saved_models/model_5_pos_token_char"


def load_model(filepath):
    return tf.keras.models.load_model(filepath)


def run():
    model = load_model(MODEL_PATH)
    print(model.summary())
    print("challenges")
