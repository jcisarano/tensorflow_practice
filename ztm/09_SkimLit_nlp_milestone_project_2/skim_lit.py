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


def examine_positional_embeddings(train_df):
    # examine data, most abstracts have fifteen or fewer lines
    # print(train_df["line_number"].value_counts())
    # train_df.line_number.plot.hist()
    # plt.show()

    # one-hot encode the line numbers
    # depth = 15 keeps each entry at 15 and most abstracts have fifteen or fewer lines
    # however, it may be worth experimenting with different values
    train_line_numbers_one_hot = tf.one_hot(train_df["line_number"].to_numpy(), depth=15)
    print(train_line_numbers_one_hot[:15], train_line_numbers_one_hot.shape)
    print(train_df["total_lines"].value_counts())
    print("98th percentile length:", np.percentile(train_df.total_lines, 98))
    train_df.total_lines.plot.hist()
    plt.show()


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


def get_positional_data_one_hot(train_df, val_df, test_df):
    train_line_numbers_one_hot = tf.one_hot(train_df["line_number"].to_numpy(), depth=15)  # 95% are 15 or less
    train_total_lines_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(), depth=20)  # 98% are 20 or less
    val_line_numbers_one_hot = tf.one_hot(val_df["line_number"].to_numpy(), depth=15)
    val_total_lines_one_hot = tf.one_hot(val_df["total_lines"].to_numpy(), depth=20)
    test_line_numbers_one_hot = tf.one_hot(test_df["line_number"].to_numpy(), depth=15)
    test_total_lines_one_hot = tf.one_hot(test_df["total_lines"].to_numpy(), depth=20)

    return train_line_numbers_one_hot, train_total_lines_one_hot, val_line_numbers_one_hot, val_total_lines_one_hot, \
           test_line_numbers_one_hot, test_total_lines_one_hot


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


def fit_pretrained_tokens_with_char_embeddings(X_train, y_train, X_val, y_val_one_hot, y_val_encoded, num_classes):
    """
        1) Create a token-level embedding model (similar to model 1
        2) Create a character-level embedding model (similar to model 3)
        3) Combine 1 and 2 using a concatenate layer
        4) Build a series of output layers
        5) Construct a model that takes token and char-level sequences as input and produces sequence label probs output
    :return:
    """

    # combine training char and token inputs and labels into one dataset set up for batching and prefetch
    train_chars = [split_chars(sentence) for sentence in X_train]
    train_char_token_data = tf.data.Dataset.from_tensor_slices((X_train, train_chars))
    train_char_token_labels = tf.data.Dataset.from_tensor_slices(y_train)
    train_char_token_dataset = tf.data.Dataset.zip((train_char_token_data, train_char_token_labels))
    train_char_token_dataset = train_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    # print(train_char_token_dataset)

    # combine validation char and token inputs and labels into one dataset configured for batching and prefetch
    val_chars = [split_chars(sentence) for sentence in X_val]
    val_char_token_data = tf.data.Dataset.from_tensor_slices((X_val, val_chars))
    val_char_token_labels = tf.data.Dataset.from_tensor_slices(y_val_one_hot)
    val_char_token_dataset = tf.data.Dataset.zip((val_char_token_data, val_char_token_labels))
    val_char_token_dataset = val_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    # print(val_char_token_dataset)

    # set up token input model
    token_inputs = layers.Input(shape=[], dtype=tf.string, name="token_input")
    pretrained_token_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                                      trainable=False,
                                                      name="universal_sentence_encoder")
    token_embedding = pretrained_token_embedding_layer(token_inputs)
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
    char_bi_lstm = layers.Bidirectional(layers.LSTM(24))(char_embeddings)  # shown in fig 1 of paper
    char_model = tf.keras.Model(inputs=char_inputs, outputs=char_bi_lstm)

    # Concatenate token and char output into input for next model
    token_char_concat = layers.Concatenate(name="token_char_concat")([token_model.output,
                                                                      char_model.output])

    # Create output layers, adding in Dropout as in section 4.2 of paper
    # dropout helps reduce overfitting
    combined_droput = layers.Dropout(rate=0.5)(token_char_concat)
    combined_dense = layers.Dense(128, activation="relu")(combined_droput)
    final_dropout = layers.Dropout(0.5)(combined_dense)
    output_layer = layers.Dense(num_classes, activation="softmax")(final_dropout)

    # Create combined model
    model = tf.keras.Model(inputs=[token_model.input, char_model.input],
                           outputs=output_layer,
                           name="model_4_token_and_char_embeddings"
                           )

    # plot the model (saves to model.png)
    # from tensorflow.keras.utils import plot_model
    # plot_model(model, show_shapes=True)

    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),  # paper uses SGD, can try that later
                  metrics=["accuracy"])

    history = model.fit(train_char_token_dataset,
                        epochs=3,  # just a few epochs for fast experimentation
                        steps_per_epoch=int(0.1 * len(train_char_token_dataset)),  # 10% of data for faster experiments
                        validation_data=val_char_token_dataset,
                        validation_steps=int(0.1 * len(val_char_token_dataset)),
                        workers=-1
                        )

    model.evaluate(val_char_token_dataset)

    pred_probs = model.predict(val_char_token_dataset)
    # print(pred_probs[:10])
    probs = np.argmax(pred_probs, axis=1)
    # print(probs[:10])
    results = calculate_results(y_val_encoded, probs)
    return model, results


def fit_pretrained_tokens_and_chars_and_position(X_train, y_train, X_val, y_val_one_hot, y_val_encoded,
                                                 train_line_numbers_one_hot,
                                                 train_total_lines_one_hot,
                                                 val_line_numbers_one_hot,
                                                 val_total_lines_one_hot, num_classes):
    """
        1) Create token-level model
        2) Create character-level model
        3) Create model for line number feature
        4) Create model for total lines feature
        5) Combine outputs of 1 & 2 using Concatenate
        6) Combine outputs of 3, 4 & 5 using Concatenate
        7) Create an output layer to accept final embedding and output label probs
        8) Combine inputs of 1, 2, 3, 4 and output (7) into tf.Keras.Model
    :return:
    """
    train_chars = [split_chars(sentence) for sentence in X_train]
    train_chars_token_pos_data = tf.data.Dataset.from_tensor_slices((train_line_numbers_one_hot,
                                                                     train_total_lines_one_hot, X_train, train_chars))
    train_chars_token_pos_labels = tf.data.Dataset.from_tensor_slices(y_train)
    train_char_token_pos_dataset = tf.data.Dataset.zip((train_chars_token_pos_data, train_chars_token_pos_labels))
    train_char_token_pos_dataset = train_char_token_pos_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    val_chars = [split_chars(sentence) for sentence in X_val]
    val_chars_token_pos_data = tf.data.Dataset.from_tensor_slices((val_line_numbers_one_hot,
                                                                   val_total_lines_one_hot, X_val, val_chars))
    val_chars_token_pos_labels = tf.data.Dataset.from_tensor_slices(y_val_one_hot)
    val_char_token_pos_dataset = tf.data.Dataset.zip((val_chars_token_pos_data, val_chars_token_pos_labels))
    val_char_token_pos_dataset = val_char_token_pos_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    # print(train_char_token_pos_dataset)
    # print(val_char_token_pos_dataset)

    # set up pretrained token model
    token_input = layers.Input(shape=[], dtype=tf.string, name="token_input_layer")
    pretrained_embedding = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                          trainable=False,
                                          name="universal_sentence_encoder")
    token_embedding = pretrained_embedding(token_input)
    token_output = layers.Dense(128, activation="relu")(token_embedding)
    token_model = tf.keras.Model(inputs=token_input, outputs=token_output)

    # set up character model
    char_input = layers.Input(shape=(1,), dtype=tf.string, name="char_input_layer")
    sent_lens = [len(sentence) for sentence in X_train]
    sent_len_95th_perc = int(np.percentile(sent_lens, 95))
    char_vectorizer = create_text_vectorizer_layer(train_chars,
                                                   sequence_len=sent_len_95th_perc,
                                                   max_vocab_len=NUM_CHAR_TOKENS)
    char_vectors = char_vectorizer(char_input)

    char_embedder = create_embedding_layer(output_dim=25,
                                           max_vocab_len=NUM_CHAR_TOKENS,
                                           mask_zero=False,
                                           name="char_embedding")
    char_embeddings = char_embedder(char_vectors)
    char_bi_lstm = layers.Bidirectional(layers.LSTM(24))(char_embeddings)
    char_model = tf.keras.Model(inputs=char_input, outputs=char_bi_lstm)

    line_num_inputs = layers.Input(shape=(15,), dtype=tf.float32, name="line_num_inputs")  # shape is size of one-hot
    x = layers.Dense(32, activation="relu")(line_num_inputs)
    line_num_model = tf.keras.Model(inputs=line_num_inputs, outputs=x)

    line_len_inputs = layers.Input(shape=(20,), dtype=tf.float32, name="line_len_inputs")  # sdhape is size of one-hot
    x = layers.Dense(32, activation="relu")(line_len_inputs)
    line_len_model = tf.keras.Model(inputs=line_len_inputs, outputs=x)

    # concatenate embeddings
    combined_embeddings = layers.Concatenate(name="token_char_hybrid_embedding")(
        [token_model.output, char_model.output])

    # add dropout layer
    combined_embeddings_with_dropout = layers.Dense(256, activation="relu")(combined_embeddings)
    combined_embeddings_with_dropout = layers.Dropout(0.5)(combined_embeddings_with_dropout)

    tribrid_embeddings = layers.Concatenate(name="char_token_positional_embedding")([line_num_model.output,
                                                                                     line_len_model.output,
                                                                                     combined_embeddings_with_dropout])

    # create output layer
    output_layer = layers.Dense(num_classes, activation="softmax", name="softmax")(tribrid_embeddings)

    # create complete model
    model = tf.keras.Model(inputs=[
        line_num_model.input, line_len_model.input, token_model.input, char_model.input
    ],
        outputs=output_layer,
        name="model_5_token_char_positional"
    )

    print(model.summary())

    # plot the model (saves to png file)
    # from tensorflow.keras.utils import plot_model
    # plot_model(model, show_shapes=True, to_file="model_5.png")

    # Compile
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),  # helps reduce overfitting
                  optimizer=tf.keras.optimizers.Adam(),  # paper uses SGD, worth a try
                  metrics=["accuracy"])

    history = model.fit(train_char_token_pos_dataset,
                        epochs=3,
                        steps_per_epoch=int(0.1*len(train_char_token_pos_dataset)),
                        validation_data=val_char_token_pos_dataset,
                        validation_steps=int(0.1*len(val_char_token_pos_dataset)),
                        workers=-1)

    # model.evaluate(val_char_token_pos_dataset)

    pred_probs = model.predict(val_char_token_pos_dataset)
    preds = tf.argmax(pred_probs, axis=1)
    results = calculate_results(y_val_encoded, preds)

    return model, results


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

    # train_dataset, val_dataset, test_dataset = format_data_for_batching(train_df["text"], train_labels_one_hot,
    #                                                                     val_df["text"], val_labels_one_hot,
    #                                                                     test_df["text"], test_labels_one_hot
    #                                                                     )

    # model_1, model_1_results = fit_conv1d(train_df["text"], train_dataset, val_dataset, val_labels_encoded,
    #                                       len(class_names))
    # print(model_1_results)

    # model_2, model_2_results = fit_model_with_USE(train_dataset, val_dataset, val_labels_encoded, len(class_names))
    # print(model_2_results)

    # Split sequence-level data splits into character-level splits
    # train_chars = [split_chars(sentence) for sentence in train_df["text"].tolist()]
    # val_chars = [split_chars(sentence) for sentence in val_df["text"].tolist()]
    # test_chars = [split_chars(sentence) for sentence in test_df["text"].tolist()]

    # examine_sentence_char_data(train_df["text"].tolist())

    # model_3, model_3_results = fit_conv1d_character_embedded(train_df["text"], train_labels_one_hot, val_df["text"],
    #                                                          val_labels_one_hot, val_labels_encoded, len(class_names))
    # print(model_3_results)

    # model_4, model_4_results = fit_pretrained_tokens_with_char_embeddings(train_df["text"],
    #                                                                       train_labels_one_hot,
    #                                                                       val_df["text"],
    #                                                                       val_labels_one_hot,
    #                                                                       val_labels_encoded,
    #                                                                       len(class_names))
    # print(model_4_results)

    train_line_numbers_one_hot, train_total_lines_one_hot, val_line_numbers_one_hot, val_total_lines_one_hot, \
    test_line_numbers_one_hot, test_total_lines_one_hot = get_positional_data_one_hot(train_df, val_df, test_df)

    # examine_positional_embeddings(train_df)

    # model_5, model_5_results = fit_pretrained_tokens_and_chars_and_position(train_df["text"], train_labels_one_hot,
    #                                                                         val_df["text"],
    #                                                                         val_labels_one_hot, val_labels_encoded,
    #                                                                         train_line_numbers_one_hot,
    #                                                                         train_total_lines_one_hot,
    #                                                                         val_line_numbers_one_hot,
    #                                                                         val_total_lines_one_hot,
    #                                                                         len(class_names))
    # print(model_5_results)

    filename = "saved_models/model_5_pos_token_char"
    # model_5.save(filename)
    model_5_loaded = tf.keras.models.load_model(filename)
    print(model_5_loaded.summary())

    val_chars = [split_chars(sentence) for sentence in val_df["text"]]
    val_chars_token_pos_data = tf.data.Dataset.from_tensor_slices((val_line_numbers_one_hot,
                                                                     val_total_lines_one_hot, val_df["text"], val_chars))
    val_chars_token_pos_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
    val_char_token_pos_dataset = tf.data.Dataset.zip((val_chars_token_pos_data, val_chars_token_pos_labels))
    val_char_token_pos_dataset = val_char_token_pos_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    pred_probs = model_5_loaded.predict(val_char_token_pos_dataset)
    probs = tf.argmax(pred_probs, axis=1)
    print(probs[:10])
    results = calculate_results(val_labels_encoded, probs)

    print(results)


    # all_model_results = pd.DataFrame({"baseline": model_0_results,
    #                                   "custom_token_embedding": model_1_results,
    #                                   "pretrained_token_embedding": model_2_results,
    #                                   "custom_char_embedding": model_3_results,
    #                                   "hybrid_char_token_embedding": model_4_results,
    #                                   "pos_char_token_embedding": model_5_results
    #                                   })
    # all_model_results = all_model_results.transpose()
    # all_model_results["accuracy"] = all_model_results["accuracy"] / 100.
    # all_model_results.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0))
    # plt.show()
    # print(all_model_results)

    # plot f1 scores
    # all_model_results.sort_values("f1", ascending=True).plot(kind="bar", figsize=(10, 7))
    # plt.show()

