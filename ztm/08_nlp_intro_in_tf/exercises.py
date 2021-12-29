import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from nlp_fundamentals import load_data, load_train_data_10_percent, calculate_results, SAVE_DIR, tokenize_text_dataset, \
    fit_rnn, fit_conv1d, fit_pretrained_feature_extraction_practice, fit_pretrained_feature_extraction, TRAIN_PATH, \
    TEST_PATH
from nlp_fundamentals import fit_dense_model
from helper_functions import create_tensorboard_callback


def fit_dense_model_sequential(X_train, y_train, X_val, y_val, X_test, max_vocab_len=10000):
    text_vectorizer = tokenize_text_dataset(X_train, X_val, X_test)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(1,), dtype=tf.string),
        text_vectorizer,
        tf.keras.layers.Embedding(
            input_dim=max_vocab_len,
            output_dim=128,
            embeddings_initializer="uniform",
            input_length=len(X_train)
        ),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ], name="model_1_dense_sequential")

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )
    # print(model.summary())

    model.fit(
        x=X_train,
        y=y_train,
        epochs=5,
        validation_data=(X_val, y_val),
        callbacks=[create_tensorboard_callback(SAVE_DIR, experiment_name="model_1_dense_sequential")]
    )

    pred_probs = model.predict(X_val)
    probs = tf.squeeze(tf.round(pred_probs))
    results = calculate_results(y_val, probs)
    print("model_1_dense_sequential results:", results)

    return model, results


def fit_rnn_sequential(X_train, y_train, X_val, y_val, X_test, max_vocab_len=10000):
    avg_sentence_len = round(sum([len(i.split()) for i in X_train]) / len(X_train))

    text_vectorizer = TextVectorization(max_tokens=max_vocab_len,  # how many words in final vocab, None means unlimited
                                        standardize="lower_and_strip_punctuation",
                                        split="whitespace",
                                        ngrams=None,
                                        output_mode="int",
                                        output_sequence_length=avg_sentence_len,  # max length of token sequence
                                        pad_to_max_tokens=False,
                                        )
    text_vectorizer.adapt(X_train)

    embedding = tf.keras.layers.Embedding(
        input_dim=max_vocab_len,
        output_dim=128,
        embeddings_initializer="uniform",
        input_length=len(X_train)
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(1,), dtype=tf.string),
        text_vectorizer,
        embedding,
        tf.keras.layers.LSTM(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ], name="model_2_LSTM_sequential")

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        validation_data=(X_val, y_val),
        # callbacks=[create_tensorboard_callback(SAVE_DIR, experiment_name="model_2_LSTM_sequential")]
        workers=-1
    )

    pred_probs = model.predict(X_val)
    preds = tf.squeeze(tf.round(pred_probs))
    results = calculate_results(y_val, preds)
    print("model_2_LSTM_sequential results:", results)


def fit_conv1d_sequential(X_train, y_train, X_val, y_val, X_test, max_vocab_len=10000):
    avg_sentence_len = round(sum(len(sent.split()) for sent in X_train) / len(X_train))
    print(avg_sentence_len)

    text_vectorizer = TextVectorization(
        max_tokens=max_vocab_len,  # how many words in final vocab, None means unlimited
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        ngrams=None,
        output_mode="int",
        output_sequence_length=avg_sentence_len,  # max length of token sequence
        pad_to_max_tokens=False,
    )
    text_vectorizer.adapt(X_train)

    embedding = tf.keras.layers.Embedding(
        input_dim=max_vocab_len,
        output_dim=128,
        embeddings_initializer="uniform",
        input_length=len(X_train)
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(1,), dtype=tf.string),
        text_vectorizer,
        embedding,
        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=5,
            strides=1,
            activation="relu",
            padding="valid",
        ),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ], name="model_5_conv_1d")

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    model.fit(
        x=X_train,
        y=y_train,
        epochs=5,
        validation_data=(X_val, y_val),
        callbacks=[create_tensorboard_callback(SAVE_DIR, experiment_name="model_5_conv_1d_sequential")]
    )

    pred_probs = model.predict(X_val)
    preds = tf.squeeze(tf.round(pred_probs))
    results = calculate_results(y_val, preds)
    print("model_5_conv_1d_sequential results:", results)


def fit_naive_bayes_ex(X_train, y_train, X_val, y_val):
    model = Pipeline([
        ("tfid", TfidfVectorizer()),
        ("clf", MultinomialNB()),
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    results = calculate_results(y_val, preds)
    print("NB 10 percent data", results)


def fit_USE_trainable(X_train, y_train, X_val, y_val, X_test):
    sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                            input_shape=[],
                                            dtype=tf.string,
                                            trainable=True)

    model = tf.keras.models.Sequential([
        sentence_encoder_layer,
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ], name="model_6_USE_trainable")

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    model.fit(
        x=X_train,
        y=y_train,
        epochs=5,
        validation_data=(X_val, y_val),
        callbacks=[create_tensorboard_callback(SAVE_DIR, experiment_name="model_6_USE_trainable")]
    )

    pred_probs = model.predict(X_val)
    preds = tf.squeeze(tf.round(pred_probs))
    results = calculate_results(y_val, preds)
    print("model_6_USE_trainable results:", results)


def run():
    print("nlp exercises")
    train_sentences, val_sentences, test_sentences, train_labels, val_labels = load_data()

    # recreate model 1 using Keras Sequential API instead of Functional API
    # fit_dense_model_sequential(train_sentences, train_labels, val_sentences, val_labels, test_sentences)
    # fit_dense_model(train_sentences, train_labels, val_sentences, val_labels, test_sentences)

    # recreate model 2 using Keras Sequential API
    # fit_rnn_sequential(train_sentences, train_labels, val_sentences, val_labels, test_sentences)
    # fit_rnn(train_sentences, train_labels, val_sentences, val_labels, test_sentences)

    # fit_conv1d_sequential(train_sentences, train_labels, val_sentences, val_labels, test_sentences)
    # fit_conv1d(train_sentences, train_labels, val_sentences, val_labels, test_sentences)

    # train NB baseline on only 10% of training data and compare to USE performance on 10% of data
    # train_sentences_10_percent, val_sentences_10_percent, \
    # test_sentences_10_percent, train_labels_10_percent, val_labels_10_percent = load_data(fraction=0.1)
    # fit_naive_bayes_ex(
    #     X_train=train_sentences_10_percent,
    #     y_train=train_labels_10_percent,
    #     X_val=val_sentences_10_percent,
    #     y_val=val_labels_10_percent
    # )
    # fit_pretrained_feature_extraction_practice(train_sentences_10_percent,
    #                                            train_labels_10_percent,
    #                                            val_sentences_10_percent,
    #                                            val_labels_10_percent)

    fit_USE_trainable(train_sentences, train_labels, val_sentences, val_labels, test_sentences)
    fit_pretrained_feature_extraction(train_sentences, train_labels, val_sentences, val_labels, test_sentences)