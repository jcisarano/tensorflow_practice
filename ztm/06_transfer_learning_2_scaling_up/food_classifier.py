"""
Large model that will classify all 101 food types in Food101 with 10% of training data

Steps to create the model:
1) Create a ModelCheckpoint callback
2) Create a data augmentation layer to build data augmentation directly into the model
3) Build a headless (no top layers) functional EfficientNetB0 backbone model (we will create our oun output model)
4) Compile
5) Feature extract for 5 full passes (5 epochs on train set and validate on 15% of test data to save time)

"""

import tensorflow as tf

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
