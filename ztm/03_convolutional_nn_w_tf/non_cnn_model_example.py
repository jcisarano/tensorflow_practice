# try image classification using same non-cnn model we used before

import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

LOCAL_SAVE_PATH: str = os.path.join("datasets", "images")


def run():
    tf.random.set_seed(42)

    # Preprocess data (normalize the image data)
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    # Set up paths to data directory
    train_dir = os.path.join(LOCAL_SAVE_PATH, "pizza_steak/train")
    test_dir = os.path.join(LOCAL_SAVE_PATH, "pizza_steak/test")

    # Import data from directories regularize it, and turn it into batches
    train_data = train_datagen.flow_from_directory(directory=train_dir,
                                                   batch_size=32,
                                                   target_size=(224, 224),
                                                   class_mode="binary",
                                                   seed=42)
    valid_data = valid_datagen.flow_from_directory(directory=test_dir,
                                                   batch_size=32,
                                                   target_size=(224, 224),
                                                   class_mode="binary",
                                                   seed=42)

    # Create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    model.fit(train_data,
              epochs=5,
              steps_per_epoch=len(train_data),
              validation_data=valid_data,
              validation_steps=len(valid_data),
              workers=-1)
