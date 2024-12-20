"""
Given a character, or sequence of characters, what is the most probable next character?

Uses GRU, but could be swapped out with LSTM

Other ideas for improvement:
    - The easiest thing you can do to improve the results is to train it for longer (try EPOCHS = 30).
    - You can also experiment with a different start string, try adding another RNN layer to improve the model's
        accuracy, or adjust the temperature parameter to generate more or less random predictions.
    - If you want the model to generate text faster the easiest thing you can do is batch the text generation. In the
        example below the model generates 5 outputs in about the same time it took to generate 1 above.
"""

import tensorflow as tf

import numpy as np
import os
import time

PATH_TO_FILE: str = "datasets/nlp/shakespeare.txt"


def load_data():
    text = open(PATH_TO_FILE, "rb").read().decode(encoding="utf-8")
    vocab = sorted(set(text))

    print(f"Length of text: {len(text)} characters")
    print(text[:250])
    print(f"{len(vocab)} unique chars")
    print(vocab)

    return text, vocab


def text_from_ids(chars_from_ids: tf.keras.layers.StringLookup, ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


def split_input_target(sequence):
    """
    Takes a sequence as input, duplictes it, and shifts it to align the input and label for each timestep.
    For any input (letter) in the sequence, the label is the next letter in the sequence
    :param sequence:
    :return:
    """
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class OneStep(tf.keras.Model):
    """
    Class to make single step predictions
    """
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # create a mask to prevent "[UNK]" from being generated
        skip_ids = self.ids_from_chars(["[UNK]"])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())]
        )
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)

        # Only use the last rpediction
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "[UNK]" form being generated
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ides to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        return predicted_chars, states


class CustomTraining(MyModel):
    """
    Custom training model allows training loop where predictions are fed back into the model to improve on mistakes
    """
    @tf.function
    def train_step(self, inputs):
        inputs, labels = inputs
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.loss(labels, predictions)
        grads = tape.gradient(loss, self.trainable_variables)  # calculate loss
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))  # calculate updates and apply

        return {'loss': loss}


def run():
    text, vocab = load_data()

    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None
    )
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
    )

    all_ids = ids_from_chars(tf.strings.unicode_split(text, "UTF-8"))
    print(all_ids)
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    for ids in ids_dataset.take(10):
        print(chars_from_ids(ids).numpy().decode("utf-8"))

    seq_length = 100
    examples_per_epoch = len(text) // (seq_length + 1)
    sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

    for seq in sequences.take(1):
        print(chars_from_ids(seq))

    for seq in sequences.take(5):
        print(text_from_ids(chars_from_ids, seq).numpy())

    dataset = sequences.map(split_input_target)

    for input_example, target_example in dataset.take(1):
        print("Input :", text_from_ids(chars_from_ids, input_example).numpy())
        print("Target :", text_from_ids(chars_from_ids, target_example).numpy())

    # Create training batches
    BATCH_SIZE: int = 64

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE: int = 64
    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024

    model = MyModel(
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units
    )

    # see example predictions
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size")

    # output as numbers
    print(model.summary())
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    print(sampled_indices)

    # convert to text
    print("Input:\n", text_from_ids(chars_from_ids, input_example_batch[0]).numpy())
    print("\nNext Char Predictions:\n", text_from_ids(chars_from_ids, sampled_indices).numpy())

    # COMPILE
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("Mean loss:        ", example_batch_mean_loss)
    print(tf.exp(example_batch_mean_loss).numpy())

    model.compile(optimizer="adam", loss=loss)

    # CREATE CHECKPOINT CALLBACK FOR SAVE
    checkpoint_dir = "models/training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True
    )

    EPOCHS = 1
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    one_step_model = OneStep(model=model, chars_from_ids=chars_from_ids, ids_from_chars=ids_from_chars)

    start = time.time()
    states = None
    next_char = tf.constant(["ROMEO:"])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
    print('\nRun time: ', end-start)

    # Batch text generation:
    start = time.time()
    states = None
    next_char = tf.constant(['ROMEO:', 'ROMEO:', 'ROMEO:', 'ROMEO:', 'ROMEO:'])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print(result, '\n\n' + '_'*80)
    print('\nRun time: ', end - start)

    # CustomTraining model example
    model = CustomTraining(
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(dataset, epochs=1)


    print("nlp text gen")

