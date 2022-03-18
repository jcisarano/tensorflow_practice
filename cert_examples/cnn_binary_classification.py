import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

PATH_TO_IMAGES: str = "datasets/images"


def run():
    tf.random.set_seed(42)

    # Preprocess data (normalize the image data)
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    # Set up paths to data directory
    train_dir = os.path.join(PATH_TO_IMAGES, "pizza_steak/train")
    test_dir = os.path.join(PATH_TO_IMAGES, "pizza_steak/test")

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

    # build CNN model to identify patterns in images (same as Tiny VGG on CNN Explainer website)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation="relu", input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, padding="valid"),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    history = model.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data),
                        workers=-1, use_multiprocessing=True)

    print("cnn_binary_class")
