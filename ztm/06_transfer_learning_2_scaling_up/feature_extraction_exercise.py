import os

import keras
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import mixed_precision

import data_utils


def create_data_augmentation():
    data_augmentation = tf.keras.models.Sequential([
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomZoom(0.2),
        preprocessing.RandomWidth(0.2),
        preprocessing.RandomHeight(0.2),
        preprocessing.RandomRotation(0.2)
    ], name="data_augmentation")
    return data_augmentation


def create_model(train_data, test_data, use_mixed_precision=False):
    if use_mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    data_aug = create_data_augmentation()
    backbone = tf.keras.applications.EfficientNetB0(include_top=False)
    backbone.trainable = False

    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
    x = data_aug(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pooling_layer_2d")(x)
    if use_mixed_precision:
        outputs = tf.keras.layers.Dense(len(train_data.class_names), activation="softmax", dtype="float32", name="output_layer")(x)
    else:
        outputs = tf.keras.layers.Dense(len(train_data.class_names), activation="softmax", name="output_layer")(x)

    model = tf.keras.models.Model(inputs, outputs)
    print(model.summary())

    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    history = model.fit(train_data, epochs=10, validation_data=test_data, validation_steps=int(0.25 * len(test_data)),
                        workers=-1)
    result = model.evaluate(test_data)
    print(result)

    return model


def run():
    train_data_all_10_percent \
        = tf.keras.preprocessing.image_dataset_from_directory(data_utils.TRAIN_DATA_PATH,
                                                              label_mode="categorical",
                                                              image_size=data_utils.IMG_SHAPE)
    test_data = tf.keras.preprocessing.image_dataset_from_directory(data_utils.TEST_DATA_PATH,
                                                                    label_mode="categorical",
                                                                    image_size=data_utils.IMG_SHAPE,
                                                                    shuffle=False)

    # create_model(train_data_all_10_percent, test_data)
    create_model(train_data_all_10_percent, test_data, use_mixed_precision=True)


