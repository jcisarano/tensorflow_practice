"""
1. Turn test data samples into tf.data Dataset for fast loading and evaluate/predict best model on test samples
2. Find most wrong predictions from test dataset
3. Make example predictions on randomized control trial abstracts from the wild, find them on PubMed.gov, e.g. search
    there for "nutrition rct" or similar. There are also a few examples in extras directory of course github

Note: Pretrained model stored in Google drive for course if desired:
    https://storage.googleapis.com/ztm_tf_course/skimlit/skimlit_tribrid_model.zip
"""
import numpy as np
import tensorflow as tf
import pandas as pd

from helper_functions import calculate_results
from skimlit_text_processors import preprocess_text_with_line_numbers, get_labels_one_hot, get_labels_int_encoded, \
    split_chars, get_positional_data_one_hot

MODEL_PATH: str = "saved_models/model_5_pos_token_char"
DATA_DIR_20K_NUM_REPL: str = "dataset/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/"
DATA_DIR_200K_NUM_REPL: str = "dataset/pubmed-rct-master/PubMed_200k_RCT_numbers_replaced_with_at_sign/"


def load_model(filepath):
    return tf.keras.models.load_model(filepath)


def run():
    # model = load_model(MODEL_PATH)
    # print(model.summary())

    test_samples = preprocess_text_with_line_numbers(filepath=DATA_DIR_20K_NUM_REPL + "test.txt")
    test_df = pd.DataFrame(test_samples)
    # test_labels_one_hot = get_labels_one_hot(test_df["target"].to_numpy().reshape(-1, 1))
    test_labels_encoded, classes = get_labels_int_encoded(test_df["target"].to_numpy())

    # test_line_numbers_one_hot, test_total_len_one_hot = get_positional_data_one_hot(test_df["line_number"],
    #                                                                                 test_df["total_lines"])

    # test_chars = [split_chars(sentence) for sentence in test_df["text"]]
    # test_chars_tokens_pos_data = tf.data.Dataset.from_tensor_slices((test_line_numbers_one_hot,
    #                                                                  test_total_len_one_hot,
    #                                                                  test_df["text"], test_chars))
    # test_chars_tokens_pos_labels = tf.data.Dataset.from_tensor_slices(test_labels_one_hot)
    # test_chars_tokens_pos_dataset = tf.data.Dataset.zip((test_chars_tokens_pos_data, test_chars_tokens_pos_labels))
    # test_chars_tokens_pos_dataset = test_chars_tokens_pos_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    # pred_probs = model.predict(test_chars_tokens_pos_dataset, verbose=1)
    # np.savetxt("saved_predictions/model_5_pred_probs.txt", pred_probs, delimiter=",")

    pred_probs = np.loadtxt("saved_predictions/model_5_pred_probs.txt", delimiter=",")
    preds = tf.argmax(pred_probs, axis=1)
    results = calculate_results(test_labels_encoded, preds)

    print(results)

    # find most wrong predictions
    pred_classes = [classes[pred] for pred in preds]
    test_df["prediction"] = pred_classes
    test_df["pred_prob"] = tf.reduce_max(pred_probs, axis=1).numpy()
    test_df["correct"] = test_df["prediction"] == test_df["target"]

    print(test_df.head(20))



