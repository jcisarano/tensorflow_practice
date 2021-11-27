import os

import keras
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing

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


def create_model(train_data, test_data):
    data_aug = create_data_augmentation()
    backbone = tf.keras.applications.EfficientNetB0(include_top=False)
    backbone.trainable = False

    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
    x = data_aug(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pooling_layer_2d")(x)
    outputs = tf.keras.layers.Dense(len(train_data.class_names), activation="softmax", name="output_layer")(x)

    model = tf.keras.models.Model(inputs, outputs)
    print(model.summary())

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

    model = create_model(train_data_all_10_percent, test_data)


