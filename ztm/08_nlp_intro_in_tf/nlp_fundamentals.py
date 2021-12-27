"""
Dataset is Kaggle intro to NLP dataset: tweets labeled as disaster/not disaster. Original source available
at Kaggle: https://www.kaggle.com/c/nlp-getting-started
"""
import io

from keras.layers import LSTM

from helper_functions import create_tensorboard_callback, plot_loss_curves, compare_histories

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
import tensorflow_hub as hub
import pandas as pd
import random
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from helper_functions import create_tensorboard_callback

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
    # print(len(train_sentences), len(train_labels), len(val_sentences), len(val_labels))
    # print(train_sentences[:10], train_labels[:10])

    return train_sentences, val_sentences, test_df["text"], train_labels, val_labels


def load_train_data_10_percent(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train_df = pd.read_csv(train_path)

    train_df_shuffled = train_df.sample(frac=1, random_state=42)
    train_10_percent = train_df_shuffled[["text", "target"]].sample(frac=0.1, random_state=42)
    train_sentences_10_percent = train_10_percent["text"].to_list()
    train_labels_10_percent = train_10_percent["target"].to_list()

    # see the new data count
    print(len(train_sentences_10_percent), len(train_labels_10_percent))

    # check the class distribution
    # it is not exactly 50/50, but is close to the original distribution
    print(train_10_percent["target"].value_counts())

    return train_sentences_10_percent, train_labels_10_percent


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
    words_in_vocab = text_vectorizer.get_vocabulary()
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
    embedding = tf.keras.layers.Embedding(input_dim=max_vocab_len,
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


def calculate_results_keras(y_true, y_pred, num_classes=2, threshold=0.5):
    a = tf.keras.metrics.Accuracy()
    a.update_state(y_true, y_pred)
    accuracy = a.result().numpy()

    p = tf.keras.metrics.Precision()
    p.update_state(y_true, y_pred)
    precision = p.result().numpy()

    r = tf.keras.metrics.Recall()
    r.update_state(y_true, y_pred)
    recall = r.result().numpy()

    # TODO: fix error in F1Score() calculation
    f1_score = None
    # f = tfa.metrics.F1Score(num_classes=num_classes, threshold=threshold)
    # f.update_state(y_true, y_pred)
    # f1_score = f.result().numpy()
    # print(f1_score)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1_score}


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
    # print(model.summary())

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
                        callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                               experiment_name="model_1_dense")])

    # print(model.evaluate(X_val, y_val))

    pred_probs = model.predict(X_val)
    # print(pred_probs.shape)
    # print(pred_probs[:10])

    preds = tf.squeeze(tf.round(pred_probs))  # convert probabilities to binary labels for evaluation
    results = calculate_results(y_true=y_val, y_pred=preds)
    print(results)

    words_in_vocab = text_vectorizer.get_vocabulary()
    print(len(words_in_vocab), words_in_vocab[:10])
    print(model.summary())
    embed_weights = model.get_layer("embedding_1").get_weights()[0]
    print(embed_weights.shape)

    """
    Visualize word embeddings with projector.tensorflow.org
    More info: https://www.tensorflow.org/text/guide/word_embeddings
    """
    save_vocab_and_weights(words_in_vocab, embed_weights)


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
                        callbacks=[create_tensorboard_callback(SAVE_DIR,
                                                               "model_2_LSTM")])

    pred_probs = model.predict(X_val)
    print(pred_probs[:10])

    preds = tf.squeeze(tf.round(pred_probs))
    print(preds[:10])

    results = calculate_results(y_val, preds)
    print(results)


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
                        callbacks=[create_tensorboard_callback(SAVE_DIR, "model_3_GRU")]
                        )

    pred_probs = model.predict(X_val)
    print(pred_probs[:10])

    preds = tf.squeeze(tf.round(pred_probs))
    print(preds[:10])

    results = calculate_results(y_val, preds)
    print(results)


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
                        callbacks=[create_tensorboard_callback(SAVE_DIR,
                                                               experiment_name="model_4_bidirectional")])

    pred_probs = model.predict(X_val)
    preds = tf.squeeze(tf.round(pred_probs))
    results = calculate_results(y_val, preds)
    print(results)


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
                        callbacks=[create_tensorboard_callback(SAVE_DIR,
                                                               experiment_name="model_5_conv_1d")])

    pred_probs = model.predict(X_val)
    preds = tf.squeeze(tf.round(pred_probs))
    results = calculate_results(y_val, preds)
    print(results)


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
                        callbacks=[create_tensorboard_callback(SAVE_DIR,
                                                               experiment_name="model_6_USE")])

    pred_probs = model.predict(X_val)
    preds = tf.squeeze(tf.round(pred_probs))
    results = calculate_results(y_val, preds)
    print(results)


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
    # fit_naive_bayes(train_sentences, train_labels, val_sentences, val_labels)
    # fit_dense_model(train_sentences, train_labels, val_sentences, val_labels, test_sentences)
    # fit_rnn(train_sentences, train_labels, val_sentences, val_labels, test_sentences)
    # fit_gru_lstm(train_sentences, train_labels, val_sentences, val_labels, test_sentences)
    # fit_bidirectional_lstm(train_sentences, train_labels, val_sentences, val_labels, test_sentences)
    # test_conv1d(train_sentences, train_labels, val_sentences, val_labels, test_sentences)
    # fit_conv1d(train_sentences, train_labels, val_sentences, val_labels, test_sentences)

    # tf_hub_test()
    # fit_pretrained_feature_extraction(train_sentences, train_labels, val_sentences, val_labels, test_sentences)

    load_train_data_10_percent()
