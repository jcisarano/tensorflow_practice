import os
import random
import string

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow_hub as hub

from pprint import pprint

from helper_functions import calculate_results

DATA_DIR_20K_NUM_REPL: str = "dataset/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/"
DATA_DIR_200K_NUM_REPL: str = "dataset/pubmed-rct-master/PubMed_200k_RCT_numbers_replaced_with_at_sign/"

NUM_CHAR_TOKENS: int = len(string.ascii_lowercase + string.digits + string.punctuation) + 2


def convert_to_panda_df(train_samples, val_samples, test_samples):
    train_df = pd.DataFrame(train_samples)
    val_df = pd.DataFrame(val_samples)
    test_df = pd.DataFrame(test_samples)

    return train_df, val_df, test_df


def get_lines(filepath):
    """
    Load text file and returns lines in that file as list

    :return:
        List of strings with one string per line from target file
    """
    with open(filepath, "r") as f:
        return f.readlines()


def split_chars(text):
    return " ".join(list(text))


def preprocess_text_with_line_numbers(filepath):
    """
    Returns list of dictionaries of abstract line data. Sample data format:
        [{
            'line_number' : 0,
            'target': 'BACKGROUND',
            'text': 'Emotional eating is associated with overeating and the development of obesity .',
            'total_lines': 11
        },
        ...
        ]
    :param filepath:
    :return:
    """
    input_lines = get_lines(filepath)
    abstract_lines = ""
    abstract_samples = []

    for line in input_lines:
        if line.startswith("###"):  # this is an id line, the start of an abstract
            abstract_id = line
            abstract_lines = ""
        elif line.isspace():  # this is an empty line, the end of an abstract
            abstract_line_split = abstract_lines.splitlines()

            # now iterate through line in a single abstract to handle and count them:
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}
                target_text_split = abstract_line.split("\t")  # splits id from text
                line_data["target"] = target_text_split[0]
                line_data["text"] = target_text_split[1].lower()
                line_data["line_number"] = abstract_line_number  # what number is this line in the abstract?
                line_data["total_lines"] = len(abstract_line_split) - 1
                abstract_samples.append(line_data)
        else:
            abstract_lines += line

    return abstract_samples


def visualize_data(train_df, val_df, test_df):
    print(train_df.head(14))

    # distribution of labels
    print(train_df.target.value_counts())

    # line length
    train_df.total_lines.plot.hist()
    plt.show()

    train_sentences = train_df["text"].tolist()
    val_sentences = val_df["text"].tolist()
    test_sentences = test_df["text"].tolist()

    print(train_sentences[:10])


def examine_sentence_data(sentences):
    sent_lens = [len(sentence.split()) for sentence in sentences]
    avg_sent_len = np.mean(sent_lens)

    print("Average sentence length:", avg_sent_len)

    plt.hist(sent_lens, bins=30)
    plt.show()

    # determine 95th percentile sentence length
    # i.e. 95% of sentences are shorter than this length
    output_sequence_len = int(np.percentile(sent_lens, 95))
    print("95% of sentences are below length ", output_sequence_len)
    print("Max sentence length: ", max(sent_lens))
    longest = sentences[np.argmax(sent_lens)]
    print("Longest sentence:", longest)


def examine_sentence_char_data(sentences):
    char_lens = [len(sentence) for sentence in sentences]
    mean_char_len = np.mean(char_lens)
    print(mean_char_len)

    plt.hist(char_lens, bins=20)
    plt.show()

    # view 95th percentile char len
    char_len_95th = int(np.percentile(char_lens, 95))
    print("95th percentile char len", char_len_95th)

    # count all keyboard characters for max token count
    import string
    alphabet = string.ascii_lowercase + string.digits + string.punctuation
    print("Alphabet count:", NUM_CHAR_TOKENS)


def create_text_vectorizer_layer(X_train, sequence_len, max_vocab_len=68000, visualize=False):
    text_vectorizer = TextVectorization(max_tokens=max_vocab_len,  # how many words in final vocab, None means unlimited
                                        output_sequence_length=sequence_len,  # max length of sequence
                                        standardize="lower_and_strip_punctuation"  # this is default value
                                        )
    text_vectorizer.adapt(X_train)

    if visualize:
        # Vectorize and visualize a random sentence
        target_sentence = random.choice(X_train)
        print(f"Text:\n{target_sentence}")
        print(f"Length:\n{len(target_sentence.split())}")
        print(f"Vectorized:\n{text_vectorizer([target_sentence])}")
        print(f"Vectorized length:\n{len(text_vectorizer([target_sentence][0]))}")

        # View vocab len and most/least common words
        print(f"\nNumber of words in vocab: {len(text_vectorizer.get_vocabulary())}")
        print(f"Most common words: {text_vectorizer.get_vocabulary()[:5]}")
        print(f"Least common words: {text_vectorizer.get_vocabulary()[-5:]}")

        # View the vectorizer configuration
        print("\nVectorizer config:")
        pprint(text_vectorizer.get_config())

    return text_vectorizer


def create_embedding_layer(max_vocab_len=68000, visualize=False, X_train=False, output_dim=128, name="token_embedding",
                           mask_zero=True):
    token_embed = layers.Embedding(input_dim=max_vocab_len,
                                   output_dim=output_dim,
                                   mask_zero=mask_zero,
                                   name=name
                                   )

    if visualize:
        import random
        text_vectorizer = create_text_vectorizer_layer(X_train)
        target_sentence = random.choice(X_train)
        target_sentence_vectorized = text_vectorizer([target_sentence])
        print(f"Text:\n{target_sentence}")
        print(f"Vectorized:\n{target_sentence_vectorized}")
        embedded_sentence = token_embed(target_sentence_vectorized)
        print(f"Embedded sentence:\n{embedded_sentence}")
        print(f"Embedded sentence shape:\n{embedded_sentence.shape}")

    return token_embed


def format_data_for_batching(X_train, y_train_one_hot,
                             X_val, y_val_one_hot,
                             X_test, y_test_one_hot):
    # Use tensorflow datasets to make data loads as fast as possible
    # see https://www.tensorflow.org/guide/data
    # and https://www.tensorflow.org/guide/data_performance
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_one_hot))
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_one_hot))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_one_hot))

    train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    return train_dataset, valid_dataset, test_dataset


def get_labels_one_hot(y_train, y_val, y_test):
    from sklearn.preprocessing import OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse=False)
    train_labels_one_hot = one_hot_encoder.fit_transform(y_train)
    val_labels_one_hot = one_hot_encoder.transform(y_val)
    test_labels_one_hot = one_hot_encoder.transform(y_test)

    return train_labels_one_hot, val_labels_one_hot, test_labels_one_hot


def get_labels_int_encode(y_train, y_val, y_test):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(y_train)
    val_labels_encoded = label_encoder.transform(y_val)
    test_labels_encoded = label_encoder.transform(y_test)

    return train_labels_encoded, val_labels_encoded, test_labels_encoded, label_encoder.classes_


def fit_naive_bayes(X_train, y_train, X_val, y_val):
    """
    Fit and train TF-IDF Multinomial Naive Bayes model as baseline for comparison to all other models.
    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :return:
    """

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    results = calculate_results(y_val, preds)

    return model, results


def fit_conv1d(X_train, train_dataset, val_dataset, y_val, num_classes):  # , y_train, X_val, y_val, num_classes):
    sent_lens = [len(sentence) for sentence in X_train]
    ninety_five_percentile_len = int(np.percentile(sent_lens, 95))
    inputs = layers.Input(shape=(1,), dtype=tf.string)
    text_vectorizer = create_text_vectorizer_layer(X_train=X_train, sequence_len=ninety_five_percentile_len)
    text_vectors = text_vectorizer(inputs)
    embedding = create_embedding_layer()
    token_embeddings = embedding(text_vectors)
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(token_embeddings)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    print(model.summary())

    # train and validate on only 10% of data during development to speed up experimentation cycle
    history = model.fit(train_dataset,
                        steps_per_epoch=int(0.1 * len(train_dataset)),
                        epochs=3,
                        validation_data=val_dataset,
                        validation_steps=int(0.1 * len(val_dataset)),
                        workers=-1
                        )

    # evaluate on whole validation set
    # print(model.evaluate(val_dataset))

    pred_probs = model.predict(val_dataset)
    preds = tf.argmax(pred_probs, axis=1)
    results = calculate_results(y_val, preds)

    return model, results


def fit_model_with_USE(train_dataset, valid_dataset, y_val, num_classes):
    tf_hub_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                            trainable=False,
                                            name="universal_encoding_layer")

    inputs = layers.Input(shape=[], dtype=tf.string)
    pretrained_embedding = tf_hub_embedding_layer(inputs)
    x = layers.Dense(128, activation="relu")(pretrained_embedding)
    output = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, output, name="model_2_USE_feature_extractor")

    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    history = model.fit(train_dataset,
                        epochs=3,
                        steps_per_epoch=int(0.1 * len(train_dataset)),
                        validation_data=valid_dataset,
                        validation_steps=int(0.1 * len(valid_dataset)))

    pred_probs = model.predict(valid_dataset)
    preds = tf.argmax(pred_probs, axis=1)
    results = calculate_results(y_val, preds)

    return model, results


def fit_conv1d_character_embedded(X_train, y_train, X_val, y_val_one_hot, y_val_encoded, num_classes):
    train_chars = [split_chars(sentence) for sentence in X_train]
    train_chars_dataset = tf.data.Dataset.from_tensor_slices((train_chars, y_train)).batch(32).prefetch(
        tf.data.AUTOTUNE)

    val_chars = [split_chars(sentence) for sentence in X_val]
    val_chars_dataset = tf.data.Dataset.from_tensor_slices((val_chars, y_val_one_hot)).batch(32).prefetch(
        tf.data.AUTOTUNE)

    # test_chars = [split_chars(sentence) for sentence in X_test]
    # test_chars_dataset = tf.data.Dataset.from_tensor_slices((test_chars, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)

    sent_lens = [len(sentence) for sentence in X_train]
    ninety_five_percentile_len = int(np.percentile(sent_lens, 95))
    char_vectorizer = create_text_vectorizer_layer(train_chars,
                                                   sequence_len=ninety_five_percentile_len,
                                                   max_vocab_len=NUM_CHAR_TOKENS)
    char_embedding = create_embedding_layer(output_dim=25,
                                            max_vocab_len=NUM_CHAR_TOKENS,
                                            name="char_embedding",
                                            mask_zero=False)

    # test_sent = random.choice(train_chars)
    # print(f"Charified text:\n{test_sent}")
    # example = char_embedding(char_vectorizer([test_sent]))
    # print(f"Vectorized and embedded:\n{example}")
    # print(f"Shape: {example.shape}")

    inputs = layers.Input(shape=(1,), dtype=tf.string)
    char_vectorized = char_vectorizer(inputs)
    char_embedded = char_embedding(char_vectorized)
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(char_embedded)
    x = layers.GlobalMaxPooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="model_3_conv1d_char_embeddings")

    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    # print(model.summary())

    history = model.fit(train_chars_dataset,
                        steps_per_epoch=int(0.1 * len(train_chars_dataset)),
                        epochs=3,
                        validation_data=val_chars_dataset,
                        validation_steps=int(0.1 * len(val_chars_dataset)),
                        workers=-1
                        )

    pred_probs = model.predict(val_chars_dataset)
    preds = tf.argmax(pred_probs, axis=1)
    print(preds)
    results = calculate_results(y_val_encoded, preds)

    return model, results


def fit_pretrained_tokens_with_char_embeddings(X_train, y_train):
    """
        1) Create a token-level embedding model (similar to model 1
        2) Create a character-level embedding model (similar to model 3)
        3) Combine 1 and 2 using a concatenate layer
        4) Build a series of output layers
        5) Construct a model that takes token and char-level sequences as input and produces sequence label probs output
    :return:
    """
    train_chars = [split_chars(sentence) for sentence in X_train]
    train_chars_dataset = tf.data.Dataset.from_tensor_slices((train_chars, y_train)).batch(32).prefetch(
        tf.data.AUTOTUNE)

    # set up token input model
    token_inputs = layers.Input(shape=[], dtype=tf.string, name="token_input")
    token_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                           trainable=False,
                                           name="universal_sentence_encoder")
    token_embedding = token_embedding_layer(token_inputs)
    token_output = layers.Dense(128, activation="relu")(token_embedding)  # output num based on paper, can increase
    token_model = tf.keras.Model(inputs=token_inputs, outputs=token_output)

    # set up character input model
    char_inputs = layers.Input(shape=[1, ], dtype=tf.string, name="char_input")
    sent_lens = [len(sentence) for sentence in X_train]
    ninety_five_percentile_len = int(np.percentile(sent_lens, 95))
    char_vectorizer = create_text_vectorizer_layer(train_chars,
                                                   sequence_len=ninety_five_percentile_len,
                                                   max_vocab_len=NUM_CHAR_TOKENS)
    char_vectors = char_vectorizer(char_inputs)

    char_embed = create_embedding_layer(output_dim=25,
                                        max_vocab_len=NUM_CHAR_TOKENS,
                                        name="char_embedding",
                                        mask_zero=False)
    char_embeddings = char_embed(char_vectors)
    char_bi_lstm = layers.Bidirectional(layers.LSTM(25))(char_embeddings)  # shown in fig 1 of paper
    char_model = tf.keras.Model(inputs=char_inputs, outputs=char_bi_lstm)

    return None, None


def parse_file(filepath):
    """
    My messed up version of the preprocess_text_with_line_numbers() task.
    :param filepath:
    :return:
    """
    lines = get_lines(filepath)
    parsed_lines = []
    for line in lines:
        line = line.replace("\n", "")
        if line.startswith("###"):
            abstract = {"line_number": 0, "total_lines": 0}
        elif len(line) == 0:
            parsed_lines.append(abstract)
            continue
        else:
            split = line.split("\t")
            abstract["target"] = split[0]
            abstract["text"] = split[1]
            abstract["total_lines"] += 1

    print(len(parsed_lines))
    print(parsed_lines[:5])

    return parsed_lines


def run():
    train_samples = preprocess_text_with_line_numbers(filepath=DATA_DIR_20K_NUM_REPL + "train.txt")
    val_samples = preprocess_text_with_line_numbers(filepath=DATA_DIR_20K_NUM_REPL + "dev.txt")
    test_samples = preprocess_text_with_line_numbers(filepath=DATA_DIR_20K_NUM_REPL + "test.txt")
    train_df, val_df, test_df = convert_to_panda_df(train_samples, val_samples, test_samples)
    # visualize_data(train_df, val_df, test_df)

    train_labels_one_hot, val_labels_one_hot, test_labels_one_hot = get_labels_one_hot(
        train_df["target"].to_numpy().reshape(-1, 1),
        val_df["target"].to_numpy().reshape(-1, 1),
        test_df["target"].to_numpy().reshape(-1, 1)
    )

    train_labels_encoded, val_labels_encoded, test_labels_encoded, class_names = get_labels_int_encode(
        train_df["target"].to_numpy(),
        val_df["target"].to_numpy(),
        test_df["target"].to_numpy(),
    )

    # model_0, model_0_results = fit_naive_bayes(train_df["text"], train_labels_encoded, val_df["text"], val_labels_encoded)
    # print(model_0_results)

    # examine_sentence_data(train_df["text"].to_numpy())

    train_dataset, val_dataset, test_dataset = format_data_for_batching(train_df["text"], train_labels_one_hot,
                                                                        val_df["text"], val_labels_one_hot,
                                                                        test_df["text"], test_labels_one_hot
                                                                        )

    # model_1, model_1_results = fit_conv1d(train_df["text"], train_dataset, val_dataset, val_labels_encoded,
    #                                       len(class_names))
    # print(model_1_results)

    # model_2, model_2_results = fit_model_with_USE(train_dataset, val_dataset, val_labels_encoded, len(class_names))
    # print(model_2_results)

    # Split sequence-level data splits into character-level splits
    train_chars = [split_chars(sentence) for sentence in train_df["text"].tolist()]
    val_chars = [split_chars(sentence) for sentence in val_df["text"].tolist()]
    test_chars = [split_chars(sentence) for sentence in test_df["text"].tolist()]

    # examine_sentence_char_data(train_df["text"].tolist())

    # model_3, model_3_results = fit_conv1d_character_embedded(train_df["text"], train_labels_one_hot, val_df["text"],
    #                                                          val_labels_one_hot, val_labels_encoded, len(class_names))
    # print(model_3_results)

    model_4, model_4_results = fit_pretrained_tokens_with_char_embeddings(train_df["text"], train_labels_one_hot)
