import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np

from nlp_fundamentals import TRAIN_PATH, TEST_PATH
from nlp_fundamentals import load_data, calculate_results


def load_train_test_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df_shuffled = train_df.sample(frac=1.0, random_state=42)

    return train_df["text"], train_df["target"], test_df["text"], test_df["id"]


def fit_USE(X_train, y_train, X_test=None, test_ids=None, num_epochs=5, X_val=None, y_val=None):
    sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                            input_shape=[],
                                            dtype=tf.string,
                                            trainable=False)
    model = tf.keras.models.Sequential([
        sentence_encoder_layer,
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ], name="best_model")

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    if X_val is None:
        model.fit(
            x=X_train,
            y=y_train,
            epochs=num_epochs,
        )
    else:
        model.fit(
            x=X_train,
            y=y_train,
            epochs=num_epochs,
            validation_data=(X_val, y_val)
        )

    if X_val is not None:
        pred_probs = model.predict(X_val)
        preds = tf.squeeze(tf.round(pred_probs))
        results = calculate_results(y_val, preds)
        print("Best model results:", results)

    if X_test is not None:
        pred_probs = model.predict(X_test)
        preds = tf.squeeze(tf.round(pred_probs))
        output = np.c_[test_ids, preds]
        np.savetxt("output/out.csv", output.astype(int), delimiter=',', fmt='%s')



def run():
    # train_sentences, val_sentences, test_sentences, train_labels, val_labels = load_data()
    # fit_USE(train_sentences, train_labels, num_epochs=5, X_val=val_sentences, y_val=val_labels)
    X_train, y_train, X_test, test_ids = load_train_test_data()
    fit_USE(X_train, y_train, X_test, num_epochs=10, test_ids=test_ids)
