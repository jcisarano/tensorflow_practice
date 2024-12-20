"""
Dataset is Kaggle intro to NLP dataset: tweets labeled as disaster/not disaster. Original source available
at Kaggle: https://www.kaggle.com/c/nlp-getting-started
"""
import io

from keras.layers import LSTM, TextVectorization
from matplotlib import pyplot as plt

# from helper_functions import create_tensorboard_callback, plot_loss_curves, compare_histories

import tensorflow as tf
# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# from tensorflow.keras.layers import Embedding
import tensorflow_hub as hub
import pandas as pd
import random
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import time

TRAIN_PATH: str = "datasets/train.csv"
TEST_PATH: str = "datasets/test.csv"

SAVE_DIR: str = "model_logs"


def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH, fraction=1.0):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df_shuffled = train_df.sample(frac=fraction, random_state=42)
    # visualize_train_data(train_df_shuffled)

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                                train_df_shuffled["target"].to_numpy(),
                                                                                test_size=0.1,
                                                                                random_state=42)
    print(len(train_sentences), len(train_labels), len(val_sentences), len(val_labels))
    # print(train_sentences[:10], train_labels[:10])
    print(train_df_shuffled["target"].value_counts())

    print(test_df.columns)

    return train_sentences, val_sentences, test_df["text"], train_labels, val_labels


def tokenize_text_dataset(train_sentences, val_sentences, test_sentences, max_vocab_len=10000):
    avg_sentence_len = round(sum([len(i.split()) for i in train_sentences]) / len(train_sentences))
    print(f"Avg sentence length: {avg_sentence_len}")

    text_vectorizer = TextVectorization(max_tokens=max_vocab_len,  # how many words in final vocab, None means unlimited
                                        standardize="lower_and_strip_punctuation",
                                        split="whitespace",
                                        ngrams=None,
                                        output_mode="int",
                                        output_sequence_length=avg_sentence_len,  # max length of token sequence
                                        pad_to_max_tokens=False,
                                        )
    text_vectorizer.adapt(train_sentences)
    # words_in_vocab = text_vectorizer.get_vocabulary()
    # print("Vocab length: ", len(words_in_vocab))
    # print("Top 5 unique words in vocabulary: ", words_in_vocab[:5])
    # print("Bottom 5 unique words in vocabulary: ", words_in_vocab[-5:])
    # sample_sentence = "There's a flood in my street"
    # print(f"Vectorized sentence '{sample_sentence}'", text_vectorizer([sample_sentence]))

    # random_sentence = random.choice(train_sentences)
    # print(f"Vectorized sentence '{random_sentence}'", text_vectorizer([random_sentence]))

    return text_vectorizer


def create_embedding_for_text_dataset(train_sentences, val_sentences, test_sentences, max_vocab_len=10000):
    """
    Convert positive integers (indexes) into dense vector of fixed size.
    Parameters we care most about for embedding layer:
    `input_dim` - size of vocabulary
    `output_dim` - size of output embedding vector, e.g. a value of 100 means each token is represented by vector of 100
    `input_length` - length of sequences passed to embedding layer
    :return:
    """
    embedding = tf.keras.layers.Embedding(
        input_dim=max_vocab_len,
        output_dim=128,
        embeddings_initializer="uniform",
        input_length=len(train_sentences)
    )
    # text_vectorizer = tokenize_text_dataset(train_sentences, val_sentences, test_sentences)
    # print(embedding)
    # random_sentence = random.choice(train_sentences)
    # print(f"Original sentence: {random_sentence}, Embedded version:", embedding(text_vectorizer([random_sentence])))

    return embedding


def visualize_train_data(train_df):
    # print(train_df.head)
    # print(train_df["text"][1])

    # dataset is relatively balanced between classes
    # if it were not, see https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    # print(train_df.target.value_counts())

    # visualize random training examples
    rand_index = random.randint(0, len(train_df) - 5)
    for row in train_df[["text", "target"]][rand_index:rand_index + 5].itertuples():
        _, text, target = row
        print(f"Target: {target}", "(disaster)" if target > 0 else "(not disaster)")
        print(f"Text: {text}")
        print("-----")


def calculate_results(y_true, y_pred):
    """
    Calculate model accuracy, precision, recall and f1 score for binary classification model

    For more info see: https://scikit-learn.org/stable/modules/model_evaluation.html
    :param y_true:
    :param y_pred:
    :return:
    """
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1_score
            }


def fit_naive_bayes(train_sentences, train_labels, val_sentences, val_labels):
    """
    This will be our baseline model for comparison of all other models. It uses sklearn's Multinomial Naive Bayes with
    TF-IDF formula to convert words to numbers.

    This is not a Deep Learning algorithm. It is common to use non-DL algorithms as a baseline because of their speed
    and also as a generally good benchmark for improvement.

    :return:
    """
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),  # convert words to numbers using TF-IDF
        ("clf", MultinomialNB())  # model the text
    ])

    model.fit(train_sentences, train_labels)
    print(model)

    score = model.score(val_sentences, val_labels)
    print(f"Baseline Naive Bayes model achieves accuracy of {score * 100:.2f}%")

    preds = model.predict(val_sentences)
    print(preds[:20])
    results = calculate_results(y_true=val_labels, y_pred=preds)
    print(results)

    return model, results


def save_vocab_and_weights(vocab, weights):
    out_v = io.open('embeddings/vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('embeddings/metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()


def fit_dense_model(X_train, y_train, X_val, y_val, X_test):
    inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)  # 1D inputs
    text_vectorizer = tokenize_text_dataset(X_train, X_val, X_test)
    x = text_vectorizer(inputs)  # turn input X into numbers
    embedding = create_embedding_for_text_dataset(X_train, X_val, X_test)
    x = embedding(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)  # seems to improve over avg pooling by 1%
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # binary output layer
    model = tf.keras.Model(inputs, outputs, name="model_1_dense")

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"]
                  )

    # print(X_train.shape)
    # print(y_train.shape)
    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=5,
                        # validation_data=(X_val, y_val),
                        # callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                        #                                        experiment_name="model_1_dense")]
                        )

    # print(model.evaluate(X_val, y_val))

    pred_probs = model.predict(X_val)
    # print(pred_probs.shape)
    # print(pred_probs[:10])

    preds = tf.squeeze(tf.round(pred_probs))  # convert probabilities to binary labels for evaluation
    results = calculate_results(y_true=y_val, y_pred=preds)
    print("model_1_dense results:", results)

    words_in_vocab = text_vectorizer.get_vocabulary()
    # print(len(words_in_vocab), words_in_vocab[:10])
    # print(model.summary())
    embed_weights = model.get_layer("embedding_1").get_weights()[0]
    # print(embed_weights.shape)

    """
    Visualize word embeddings with projector.tensorflow.org
    More info: https://www.tensorflow.org/text/guide/word_embeddings
    """
    save_vocab_and_weights(words_in_vocab, embed_weights)

    return results


def fit_rnn(X_train, y_train, X_val, y_val, X_test):
    """
    RNNs are useful for sequence data, e.g. text strings.

    RNN uses representation of a previous input to aid representation of later input.
    RNN Structure:
    Input (text) -> Tokenize -> Embedding -> Layers (RNN) -> Output (label probability
    LSTM = Long short term memory
    :return:
    """
    inputs = tf.keras.layers.Input(shape=(1,), dtype="string")
    text_vectorizer = tokenize_text_dataset(train_sentences=X_train, val_sentences=X_val, test_sentences=X_test)
    x = text_vectorizer(inputs)
    embedding = create_embedding_for_text_dataset(X_train, X_val, X_test)
    x = embedding(x)
    # x = tf.keras.layers.LSTM(units=64, return_sequences=True)(x)  # when stacking RNN cells, return_sequences must be true
    x = tf.keras.layers.LSTM(units=64, activation="relu")(x)
    # x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="model_2_LSTM")

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    history = model.fit(X_train,
                        y_train,
                        epochs=5,
                        validation_data=(X_val, y_val),
                        # callbacks=[create_tensorboard_callback(SAVE_DIR,
                        #                                        "model_2_LSTM")]
                        )

    pred_probs = model.predict(X_val)
    print(pred_probs[:10])

    preds = tf.squeeze(tf.round(pred_probs))
    print(preds[:10])

    results = calculate_results(y_val, preds)
    print(results)

    return results


def fit_gru_lstm(X_train, y_train, X_val, y_val, X_test):
    """
    Gated Recurrent Unit

    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :param X_test:
    :return:
    """
    inputs = tf.keras.layers.Input(shape=(1,), dtype="string")
    text_vectorizer = tokenize_text_dataset(train_sentences=X_train, val_sentences=X_val, test_sentences=X_test)
    x = text_vectorizer(inputs)
    embedding = create_embedding_for_text_dataset(train_sentences=X_train, val_sentences=X_val, test_sentences=X_test)
    x = embedding(x)
    x = tf.keras.layers.GRU(64)(x)
    # x = tf.keras.layers.GRU(64, return_sequences=True)(x)
    # x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    # x = tf.keras.layers.GRU(64)(x)
    # x = tf.keras.layers.Dense(64, activation="relu")(x)
    # x = tf.keras.layers.GlobalMaxPooling1D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="model_3_GRU")

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    history = model.fit(X_train,
                        y_train,
                        epochs=5,
                        validation_data=(X_val, y_val),
                        # callbacks=[create_tensorboard_callback(SAVE_DIR, "model_3_GRU")]
                        )

    pred_probs = model.predict(X_val)
    print(pred_probs[:10])

    preds = tf.squeeze(tf.round(pred_probs))
    print(preds[:10])

    results = calculate_results(y_val, preds)
    print(results)

    return results


def fit_bidirectional_lstm(X_train, y_train, X_val, y_val, X_test):
    inputs = tf.keras.layers.Input(shape=(1,), dtype="string")
    text_vectorizer = tokenize_text_dataset(X_train, X_val, X_test)
    x = text_vectorizer(inputs)
    embedding = create_embedding_for_text_dataset(X_train, X_val, X_test)
    x = embedding(x)
    # x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    # x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="model_4_bidirectional")
    print(model.summary())

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    history = model.fit(X_train,
                        y_train,
                        epochs=5,
                        validation_data=(X_val, y_val),
                        # callbacks=[create_tensorboard_callback(SAVE_DIR, experiment_name="model_4_bidirectional")]
                        )

    pred_probs = model.predict(X_val)
    preds = tf.squeeze(tf.round(pred_probs))
    results = calculate_results(y_val, preds)
    print(results)

    return results


def test_conv1d(X_train, y_train, X_val, y_val, X_test):
    inputs = tf.keras.layers.Input(shape=(1,), dtype="string")
    text_vectorizer = tokenize_text_dataset(X_train, X_val, X_test)
    embedding = create_embedding_for_text_dataset(X_train, X_val, X_test)
    embedding_test = embedding(text_vectorizer(["this is a test sentence"]))

    conv_1d = tf.keras.layers.Conv1D(filters=32,  # num hidden units in layer feature vector
                                     kernel_size=5,  # looks at this many tokens at a time, aka n-gram size
                                     activation="relu",
                                     padding="valid",  # output will likely be smaller than input shape, no padding
                                     # padding="same"  # zero padding added if needed to maintain same size output
                                     )
    conv_1d_output = conv_1d(embedding_test)

    max_pool = tf.keras.layers.GlobalMaxPooling1D()  # max value for each token (of 15) in feature vector
    max_pool_output = max_pool(conv_1d_output)  # gets feature w/highest value, i.e. the most important one

    print(embedding_test.shape, conv_1d_output.shape, max_pool_output.shape)


def fit_conv1d(X_train, y_train, X_val, y_val, X_test):
    """
    Typical structure for Conv1D model for text sequences:
        Inputs (text) -> Tokenization -> Embedding -> Conv1D layer(s) (typically Conv1D + pooling) -> Output probs
    :return:
    """

    inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    text_vectorizer = tokenize_text_dataset(X_train, X_val, X_test)
    x = text_vectorizer(inputs)
    embedding = create_embedding_for_text_dataset(X_train, X_val, X_test)
    x = embedding(x)

    x = tf.keras.layers.Conv1D(filters=64,
                               kernel_size=5,
                               strides=1,
                               activation="relu",
                               padding="valid")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    # x = tf.keras.layers.Dense(64, activation="relu")(x)  # another possible layer
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="model_5_conv_1d")

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    print(model.summary())

    history = model.fit(X_train,
                        y_train,
                        epochs=5,
                        validation_data=(X_val, y_val),
                        # callbacks=[create_tensorboard_callback(SAVE_DIR, experiment_name="model_5_conv_1d")]
                        )

    pred_probs = model.predict(X_val)
    preds = tf.squeeze(tf.round(pred_probs))
    results = calculate_results(y_val, preds)
    print("model_5_conv_1d results:", results)

    return results


def tf_hub_test():
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embeddings = embed([
        "There's a flood in my street!",
        "When you pass a sentence to the universal encoder, it returns a tensor full of numbers."
    ])

    print(embeddings)


def fit_pretrained_feature_extraction(X_train, y_train, X_val, y_val, X_test):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                            input_shape=[],
                                            dtype=tf.string,
                                            trainable=False,
                                            name="USE")

    model = tf.keras.Sequential([
        sentence_encoder_layer,
        # tf.keras.layers.Dense(256, activation="relu"),
        # tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")
    ], name="model_6_USE")

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    print(model.summary())

    history = model.fit(X_train,
                        y_train,
                        epochs=5,
                        validation_data=(X_val, y_val),
                        # callbacks=[create_tensorboard_callback(SAVE_DIR, experiment_name="model_6_USE")]
                        )

    pred_probs = model.predict(X_val)
    preds = tf.squeeze(tf.round(pred_probs))
    results = calculate_results(y_val, preds)
    print(results)

    return model, results


def fit_pretrained_feature_extraction_practice(X_train, y_train, X_val, y_val):
    sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                            input_shape=[],
                                            dtype=tf.string,
                                            trainable=False,
                                            name="USE")

    model = tf.keras.Sequential([
        sentence_encoder_layer,
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")
    ], name="model_7_USE_10_percent")

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    print(model.summary())

    history = model.fit(X_train,
                        y_train,
                        epochs=5,
                        validation_data=(X_val, y_val),
                        # callbacks=[create_tensorboard_callback(SAVE_DIR, experiment_name="model_7_USE_10_percent")]
                        )

    pred_probs = model.predict(X_val)
    preds = tf.squeeze(tf.round(pred_probs))
    results = calculate_results(y_val, preds)
    print("model_7_USE_10_percent results:", results)

    return results


def load_and_fit_pretrained_model(X_train, X_test, X_val, y_train, y_val):
    # load a pretrained model for the next section, so it matches the course performance
    model = tf.keras.models.load_model("saved_models/08_model_6_USE_feature_extractor")
    # print(model_6_pretrained.evaluate(val_sentences, val_labels))
    pred_probs = model.predict(X_val)
    preds = tf.squeeze(tf.round(pred_probs))
    # print(model_6_pretrained_preds)

    val_df = pd.DataFrame({"text": X_val,
                           "target": y_val,
                           "pred": preds,
                           "pred_prob": tf.squeeze(pred_probs)})
    # print(val_df)

    # create a new dataframe where pred != target and sort so that the worst are at the top
    most_wrong = val_df[val_df["target"] != val_df["pred"]].sort_values("pred_prob", ascending=False)
    # print(most_wrong)
    # for row in most_wrong[:10].itertuples():
    #     _, text, target, pred, pred_prob = row
    #     print(f"Target: {target}, Pred: {pred}, Prob: {pred_prob}")
    #     print(f"Text:\n{text}\n")
    #     print("----\n")

    # for row in most_wrong[-10:# ].itertuples():
    #     _, text, target, pred, pred_prob = row
    #     print(f"Target: {target}, Pred: {pred}, Prob: {pred_prob}")
    #     print(f"Text:\n{text}\n")
    #     print("----\n")

    # make predictions on test dataset:
    model_6_pretrained_pred_probs_test = model.predict(X_test)
    model_6_pretrained_preds_test = tf.squeeze(tf.round(model_6_pretrained_pred_probs_test))
    test_df = pd.DataFrame({"text": X_test,
                            "pred": model_6_pretrained_preds_test,
                            "pred_prob": tf.squeeze(model_6_pretrained_pred_probs_test)})

    # print(test_df[:10])

    # print out 10 random predictions
    # test_samples = random.sample(test_df["text"].to_list(), 10)
    # for sample in test_samples:
    #     pred_prob = tf.squeeze(model_6_pretrained.predict([sample]))
    #     pred = tf.round(pred_prob)
    #     print(f"Pred: {int(pred)}, Prob: {pred_prob}")
    #     print(f"\nText: {sample}\n")
    #     print("-----\n")

    results = calculate_results(y_val, preds)
    return model, results


def save_load_model_as_hdf5(model, X_val, y_val):
    """
    Saving and loading a trained model.
    There are two main formats: hdf5 and SavedModel format (which is default for TensorFlow)
    """
    model.save("saved_models/model_6.h5")  # saves as hdf5

    # formatting to load model with custom Hub layer (required when using hdf5 format)
    loaded_model_6 = tf.keras.models.load_model("saved_models/model_6.h5",
                                                custom_objects={"KerasLayer": hub.KerasLayer})
    print(loaded_model_6.summary())
    print(loaded_model_6.evaluate(X_val, y_val))

    model.save("saved_models/model_6_SavedModel_format")
    loaded_model_6_SavedModel_format = tf.keras.models.load_model("saved_models/model_6_SavedModel_format")
    print(loaded_model_6_SavedModel_format.summary())
    print(loaded_model_6_SavedModel_format.evaluate(X_val, y_val))


def pandas_plot(results_model_0_naive_bayes, results_model_1_dense, results_model_2_rnn, results_model_3_gru,
                results_model_4_bidirectional, results_model_5_conv1d, results_model_6_use,
                results_model_7_use_10_percent):
    all_model_results = pd.DataFrame({"0_baseline": results_model_0_naive_bayes,
                                      "1_simple_dense": results_model_1_dense,
                                      "2_lstm": results_model_2_rnn,
                                      "3_gru": results_model_3_gru,
                                      "4_bidirectional": results_model_4_bidirectional,
                                      "5_conv1d": results_model_5_conv1d,
                                      "6_use_encoder": results_model_6_use,
                                      "7_use_encoder_10_percent": results_model_7_use_10_percent}).transpose()
    all_model_results["accuracy"] = all_model_results["accuracy"] / 100.
    print(all_model_results)
    all_model_results.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0))
    plt.show()

    all_model_results.sort_values("f1", ascending=False)["f1"].plot(kind="bar", figsize=(10, 7))
    plt.show()


def pred_timer(model, samples):
    """
    Time how long it takes the model to make a prediction.
    :param model:
    :param samples:
    :return:
    """
    start_time = time.perf_counter()  # gets start time
    model.predict(samples)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    time_per_pred = total_time / len(samples)
    return total_time, time_per_pred


def run():
    print("nlp fundies")
    # load the data:
    train_sentences, val_sentences, test_sentences, train_labels, val_labels = load_data()

    # Convert the text data to numbers because NN cannot handle text
    # Make sure the tensors are the right shape (may need to pad sequences to a standard length)
    # tokenize_text_dataset(train_sentences, val_sentences, test_sentences)
    create_embedding_for_text_dataset(train_sentences, val_sentences, test_sentences)

    """
    Series of modeling experiments
    Model 0: Naive Bayes will be the baseline
    Model 1: Feed-Forward neural network (dense model)
    Model 2: LSTM (RNN)
    Model 3: GRU (RNN)
    Model 4: Bidirectional-LSTM (RNN)
    Model 5: 1D Convolutional Neural Network (CNN)
    Model 6: TensorFlow Hub Pretrained Feature Extractor (transfer learning for NLP)
    Model 7: Same as model 6 with 10% of training data

    Standard tensorflow modeling steps:
    1) Create a model
    2) Build the model
    3) Fit the model
    4) Evaluate the model
    """
    model_0, results_model_0_naive_bayes = fit_naive_bayes(train_sentences, train_labels, val_sentences, val_labels)
    # results_model_1_dense = fit_dense_model(train_sentences, train_labels, val_sentences, val_labels, test_sentences)
    # results_model_2_rnn = fit_rnn(train_sentences, train_labels, val_sentences, val_labels, test_sentences)
    # results_model_3_gru = fit_gru_lstm(train_sentences, train_labels, val_sentences, val_labels, test_sentences)
    # results_model_4_bidirectional = fit_bidirectional_lstm(train_sentences, train_labels, val_sentences, val_labels,
    #                                                        test_sentences)
    # test_conv1d(train_sentences, train_labels, val_sentences, val_labels, test_sentences)
    # results_model_5_conv1d = fit_conv1d(train_sentences, train_labels, val_sentences, val_labels, test_sentences)

    # tf_hub_test()
    # model_6, results_model_6_use = fit_pretrained_feature_extraction(train_sentences, train_labels, val_sentences,
    #                                                                  val_labels,
    #                                                                  test_sentences)

    # X_train_10_percent, y_train_10_percent = load_train_data_10_percent()
    # train_sentences_10_percent, val_sentences_10_percent, \
    # test_sentences_10_percent, train_labels_10_percent, val_labels_10_percent = load_data(fraction=0.1)
    # results_model_7_use_10_percent = fit_pretrained_feature_extraction_practice(train_sentences_10_percent,
    #                                                                             train_labels_10_percent,
    #                                                                             val_sentences_10_percent,
    #                                                                             val_labels_10_percent)

    # pandas_plot(results_model_0_naive_bayes, results_model_1_dense, results_model_2_rnn, results_model_3_gru,
    #                 results_model_4_bidirectional, results_model_5_conv1d, results_model_6_use,
    #                 results_model_7_use_10_percent)

    """
    tensorboard dev upload --logdir .\model_logs\ --name "NLP Modeling Experiments" --description "Comparing multiple models' performance on Kaggle disaster t
weets dataset" --one_shot 
    Uploaded experiments to tensorboard at https://tensorboard.dev/experiment/LmyqWX86TLODJrlcqU4bdw/
    Also see https://wandb.ai/site for more visualization tools
    """

    model_6_pretrained, model_6_pretrained_results = load_and_fit_pretrained_model(train_sentences,
                                                                                   test_sentences,
                                                                                   val_sentences,
                                                                                   train_labels,
                                                                                   val_labels)

    model_6_total_time, model_6_time_per_pred = pred_timer(model_6_pretrained, val_sentences)
    print(f"Model 6 total time: {model_6_total_time}, Time per prediction: {model_6_time_per_pred}")

    baseline_total_time, baseline_time_per_pred = pred_timer(model_0, val_sentences)
    print(f"Baseline total time: {baseline_total_time}, Time per prediction: {baseline_time_per_pred}")

    # plot time to predict vs f1 score
    # baseline is 3x faster than model 6, while model 6 is 3% more accurate
    plt.figure(figsize=(10, 7))
    plt.scatter(baseline_time_per_pred, results_model_0_naive_bayes["f1"], label="baseline")
    plt.scatter(model_6_time_per_pred, model_6_pretrained_results["f1"], label="tf_hub_sentence_encoder")
    plt.legend()
    plt.title("F1 score versus time per prediction")
    plt.xlabel("Time per prediction")
    plt.ylabel("F1 score")
    plt.show()
