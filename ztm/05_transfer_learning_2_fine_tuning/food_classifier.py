# food classifier using pretrained model from tf.keras.application
# https://www.tensorflow.org/api_docs/python/tf/keras/applications
import os
import tensorflow as tf

from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir

LOCAL_SAVE_PATH: str = os.path.join("datasets", "images")
LOCAL_DATA_PATH: str = os.path.join("datasets", "images/10_food_classes_10_percent")
TRAIN_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "train")
TEST_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "test")

IMG_SIZE: int = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE: int = 32


def transfer_learning_functional_api(train_data, test_data):
    # 1. Create the base model with tf.keras.applications
    model = tf.keras.applications.EfficientNetB0(include_top=False)

    # 2. Freeze the base model (underlying, pre-trained patterns arent changed during training)
    model.trainable = False

    # 3. Create inputs
    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")

    # 4. If using a model like ResNet50V2, you will need to normalize input
    # EfficientNet does not need this
    # x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)

    # 5. pass the inputs to the base model
    x = model(inputs)
    print(f"Shape after passing inputs through base model: {x.shape}")

    # 6. Average pool the outputs of base model
    # Aggregates all the most important info and reduces computations
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    print(f"Shape after GlobalAveragePooling2D: {x.shape}")

    # 7. Create the output activation layer
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_layer")(x)

    # 8. Combine inputs with outputs into model
    model_0 = tf.keras.Model(inputs, outputs)

    # 9. Compile the model
    model_0.compile(loss="categorical_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

    # 10. Fit the model and save its history
    model_0.fit(train_data,
                epochs=5,
                steps_per_epoch=len(train_data),
                validation_data=test_data,
                validation_steps=len(test_data),
                workers=-1)



def run():
    # unzip_data("10_food_classes_10_percent.zip")
    # walk_through_dir(LOCAL_DATA_PATH)

    # returns BatchDataset, with batch size 32
    train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=TRAIN_DATA_PATH,
                                                                                label_mode="categorical",
                                                                                image_size=IMG_SHAPE,
                                                                                batch_size=BATCH_SIZE)
    test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=TEST_DATA_PATH,
                                                                    label_mode="categorical",
                                                                    image_size=IMG_SHAPE,
                                                                    batch_size=BATCH_SIZE)

    # for images, labels in train_data_10_percent.take(1):
    #    print(images, labels)

    transfer_learning_functional_api(train_data_10_percent, test_data)
