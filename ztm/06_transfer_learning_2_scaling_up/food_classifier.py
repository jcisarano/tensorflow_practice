"""
Large model that will classify all 101 food types in Food101 with 10% of training data

Steps to create the model:
1) Create a ModelCheckpoint callback
2) Create a data augmentation layer to build data augmentation directly into the model
3) Build a headless (no top layers) functional EfficientNetB0 backbone model (we will create our oun output model)
4) Compile
5) Feature extract for 5 full passes (5 epochs on train set and validate on 15% of test data to save time)

"""
import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

import data_utils


def run():
    train_data_all_10_percent \
        = tf.keras.preprocessing.image_dataset_from_directory(data_utils.TRAIN_DATA_PATH,
                                                              label_mode="categorical",
                                                              image_size=data_utils.IMG_SHAPE)
    test_data = tf.keras.preprocessing.image_dataset_from_directory(data_utils.TEST_DATA_PATH,
                                                                    label_mode="categorical",
                                                                    image_size=data_utils.IMG_SHAPE,
                                                                    shuffle=False)

    # Create checkpoint callback
    checkpoint_path = "101_classes_10_percent_data_model_checkpoint"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                             save_weights_only=True,
                                                             monitor="val_accuracy",
                                                             save_best_only=True)

    data_augmentation = Sequential([
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.2),
        preprocessing.RandomHeight(0.2),
        preprocessing.RandomWidth(0.2),
        preprocessing.RandomZoom(0.2),
        # preprocessing.Rescale(1/255.) # rescale only if model doesn't include scaling
    ], name="data_augmentation")

    # create base model and freeze its layers
    backbone = tf.keras.applications.EfficientNetB0(include_top=False)
    backbone.trainable = False

    # create trainable top layer architecture
    inputs = layers.Input(shape=(224, 224, 3), name="input_layer")
    x = data_augmentation(inputs)
    x = backbone(x, training=False)  # puts base model in inference mode, so frozen weights will stay frozen
    x = layers.GlobalAveragePooling2D(name="global_avg_pooling_2d")(x)
    outputs = layers.Dense(len(train_data_all_10_percent.class_names),
                           activation="softmax",
                           name="output_layer")(x)

    model = keras.Model(inputs, outputs)



