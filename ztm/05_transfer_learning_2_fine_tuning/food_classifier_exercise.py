import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import data_utils as du


def print_model_info(model):
    print(model)
    for index, layer in enumerate(model.layers):
        print(index, layer.name, layer.trainable)


def run():
    # load ten percent dataset
    train_data = tf.keras.preprocessing.image_dataset_from_directory(du.TRAIN_DATA_PATH,
                                                                                label_mode="categorical",
                                                                                image_size=du.IMG_SHAPE)
    test_data = tf.keras.preprocessing.image_dataset_from_directory(du.TEST_DATA_PATH,
                                                                 label_mode="categorical",
                                                                 image_size=du.IMG_SHAPE)

    base_model = tf.keras.applications.EfficientNetB0(include_top=False)
    base_model.trainable = False
    # print_model_info(base_model)

    input_shape = (du.IMG_SIZE, du.IMG_SIZE, 3)
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pooling_2d")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="dense_output_layer")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    print_model_info(model)


