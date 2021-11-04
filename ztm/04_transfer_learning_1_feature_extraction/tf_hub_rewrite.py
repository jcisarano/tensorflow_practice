# rewrite previous models following tutorial at https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub
import os

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

import tensorflow_hub as hub

import matplotlib.pylab as plt

EFFICIENTNET_URL: str = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
RESNET_URL: str = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
MOBILENET_URL: str = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
MOBILENET_V2: str = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
INCEPTION_V3: str = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

IMAGE_SHAPE = (224, 224)

LOCAL_SAVE_PATH: str = os.path.join("datasets", "images")
LOCAL_DATA_PATH: str = os.path.join("datasets", "images/10_food_classes_10_percent")
TRAIN_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "train")
TEST_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "test")


def create_model(model_path, num_classes: int = 10, input_shape=IMAGE_SHAPE):
    model = tf.keras.Sequential([
        hub.KerasLayer(model_path, input_shape=input_shape + (3,), trainable=False, name="feature_extractor_layer"),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer")
    ])

    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

    return model


def load_and_prep_data(train_dir=TRAIN_DATA_PATH, test_dir=TEST_DATA_PATH, batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1 / 255.)
    test_datagen = ImageDataGenerator(rescale=1 / 255.)

    train_data = train_datagen.flow_from_directory(train_dir,
                                                   batch_size=batch_size,
                                                   target_size=IMAGE_SHAPE,
                                                   class_mode="categorical")

    test_data = test_datagen.flow_from_directory(test_dir,
                                                 batch_size=batch_size,
                                                 target_size=IMAGE_SHAPE,
                                                 class_mode="categorical")
    return train_data, test_data


def run():
    train_data, test_data = load_and_prep_data()

    model = create_model(EFFICIENTNET_URL)
    print(model.summary())

    model.fit(train_data,
              epochs=5,
              steps_per_epoch=len(train_data),
              validation_data=test_data,
              validation_steps=len(test_data),
              workers=-1)
