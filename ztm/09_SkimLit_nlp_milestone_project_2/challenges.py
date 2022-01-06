"""
1. Turn test data samples into tf.data Dataset for fast loading and evaluate/predict best model on test samples
2. Find most wrong predictions from test dataset
3. Make example predictions on randomized control trial abstracts from the wild, find them on PubMed.gov, e.g. search
    there for "nutrition rct" or similar. There are also a few examples in extras directory of course github

Note: Pretrained model stored in Google drive for course if desired:
    https://storage.googleapis.com/ztm_tf_course/skimlit/skimlit_tribrid_model.zip
"""
import string

import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow_hub as hub

from helper_functions import calculate_results
from skimlit_text_processors import preprocess_text_with_line_numbers, get_labels_one_hot, get_labels_int_encoded, \
    split_chars, get_positional_data_one_hot

MODEL_PATH: str = "saved_models/model_5_pos_token_char"
DATA_DIR_20K_NUM_REPL: str = "dataset/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/"
DATA_DIR_200K_NUM_REPL: str = "dataset/pubmed-rct-master/PubMed_200k_RCT_numbers_replaced_with_at_sign/"

CHECKPOINT_PATH: str = "model_checkpoints/cp.ckpt"


def load_model(filepath):
    return tf.keras.models.load_model(filepath)


def find_most_wrong(test_df, preds, pred_probs, classes):
    # find most wrong predictions
    pred_classes = [classes[pred] for pred in preds]
    test_df["prediction"] = pred_classes
    test_df["pred_prob"] = tf.reduce_max(pred_probs, axis=1).numpy()
    test_df["correct"] = test_df["prediction"] == test_df["target"]

    most_wrong = test_df[test_df["correct"] == False].sort_values("pred_prob", ascending=False)[:100]
    print(most_wrong)

    for row in most_wrong[0:10].itertuples():
        _, target, text, line_num, total_lines, pred, pred_prob, _ = row
        print(
            f"Target: {target}, Pred:{pred}, Prob: {pred_prob}, Line number: {line_num}, Total_lines: {total_lines}\n")
        print(f"Text:\n{text}")
        print("-----\n")


def create_model_with_callbacks(X_train, train_chars, train_char_token_pos_dataset, val_char_token_pos_dataset,
                                num_classes):
    # token model
    token_input = layers.Input(shape=[], dtype=tf.string, name="token_input_layer")
    pretrained_embedding = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                          trainable=False,
                                          name="universal_sentence_encoder")
    token_embedding = pretrained_embedding(token_input)
    token_output = layers.Dense(128, activation="relu")(token_embedding)
    token_model = tf.keras.Model(inputs=token_input, outputs=token_output)

    # char model
    NUM_CHAR_TOKENS: int = len(string.ascii_lowercase + string.digits + string.punctuation) + 2
    sent_len = [len(sentence) for sentence in X_train]
    seq_len = int(np.percentile(sent_len, 95))
    char_input = layers.Input(shape=(1,), dtype=tf.string, name="char_input_layer")
    char_vectorizer = TextVectorization(max_tokens=NUM_CHAR_TOKENS,
                                        output_sequence_length=seq_len,
                                        standardize="lower_and_strip_punctuation"
                                        )
    char_vectorizer.adapt(train_chars)
    char_vectors = char_vectorizer(char_input)

    char_embedder = layers.Embedding(input_dim=NUM_CHAR_TOKENS,
                                     output_dim=25,
                                     mask_zero=False,
                                     name="char_embedding"
                                     )
    char_embedding = char_embedder(char_vectors)
    char_bi_lstm = layers.Bidirectional(layers.LSTM(24))(char_embedding)
    char_model = tf.keras.Model(inputs=char_input, outputs=char_bi_lstm)

    line_num_inputs = layers.Input(shape=(15,), dtype=tf.float32, name="line_num_inputs")
    x = layers.Dense(32, activation="relu")(line_num_inputs)
    line_num_model = tf.keras.Model(inputs=line_num_inputs, outputs=x)

    line_len_inputs = layers.Input(shape=(20,), dtype=tf.float32, name="line_len_inputs")
    x = layers.Dense(32, activation="relu")(line_len_inputs)
    line_len_model = tf.keras.Model(inputs=line_len_inputs, outputs=x)

    combined_embeddings = layers.Concatenate(name="token_char_hybrid_embedding")([token_model.output,
                                                                                  char_model.output])

    combined_embeddings_with_dropout = layers.Dense(256, activation="relu")(combined_embeddings)
    combined_embeddings_with_dropout = layers.Dropout(0.5)(combined_embeddings_with_dropout)

    tribrid_embedding = layers.Concatenate(name="char_token_positional_embedding")([line_num_model.output,
                                                                                    line_len_model.output,
                                                                                    combined_embeddings_with_dropout])

    output = layers.Dense(num_classes, activation="softmax", name="softmax_output")(tribrid_embedding)
    model = tf.keras.Model(inputs=[line_num_model.input, line_len_model.input, token_model.input, char_model.input],
                           outputs=output,
                           name="model_5_with_checkpoint_and_early_stop")

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=1,
        restore_best_weights=True
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH,
                                                          monitor="val_accuracy",
                                                          save_best_only=True,
                                                          save_weights_only=True,
                                                          verbose=0)

    model.fit(train_char_token_pos_dataset,
              epochs=3,
              steps_per_epoch=int(0.1*len(train_char_token_pos_dataset)),
              validation_data=val_char_token_pos_dataset,
              validation_steps=int(0.1*val_char_token_pos_dataset),
              callbacks=[early_stopping, model_checkpoint],
              workers=-1
              )

    return model

def run():
    # model = load_model(MODEL_PATH)
    # print(model.summary())

    test_samples = preprocess_text_with_line_numbers(filepath=DATA_DIR_20K_NUM_REPL + "test.txt")
    test_df = pd.DataFrame(test_samples)
    test_labels_one_hot = get_labels_one_hot(test_df["target"].to_numpy().reshape(-1, 1))
    test_labels_encoded, classes = get_labels_int_encoded(test_df["target"].to_numpy())

    test_line_numbers_one_hot, test_total_len_one_hot = get_positional_data_one_hot(test_df["line_number"],
                                                                                    test_df["total_lines"])

    test_chars = [split_chars(sentence) for sentence in test_df["text"]]
    test_chars_tokens_pos_data = tf.data.Dataset.from_tensor_slices((test_line_numbers_one_hot,
                                                                     test_total_len_one_hot,
                                                                     test_df["text"], test_chars))
    test_chars_tokens_pos_labels = tf.data.Dataset.from_tensor_slices(test_labels_one_hot)
    test_chars_tokens_pos_dataset = tf.data.Dataset.zip((test_chars_tokens_pos_data, test_chars_tokens_pos_labels))
    test_chars_tokens_pos_dataset = test_chars_tokens_pos_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    # pred_probs = model.predict(test_chars_tokens_pos_dataset, verbose=1)
    # np.savetxt("saved_predictions/model_5_pred_probs.txt", pred_probs, delimiter=",")

    # pred_probs = np.loadtxt("saved_predictions/model_5_pred_probs.txt", delimiter=",")
    # preds = tf.argmax(pred_probs, axis=1)
    # results = calculate_results(test_labels_encoded, preds)

    # print(results)

    # find_most_wrong(test_df, preds, pred_probs, classes)

    model = create_model_with_callbacks(test_df["text"], test_chars, len(classes))
    print(model.summary())


