import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random

import data_utils as du
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir


def set_up_data_aug():
    # do data augmentation
    # tf.keras.layers.experimental.preprocessing has data augmentation features
    # Using Sequential() here to pass to functional API
    # This will add data augmentation to model, so they run on the gpu and become part of the model,
    # so a saved model will include the preprocessing steps
    # this preprocessing is only used during training, not during evaluation
    data_augmentation = keras.Sequential([
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.2),
        preprocessing.RandomZoom(0.2),
        preprocessing.RandomHeight(0.2),
        preprocessing.RandomWidth(0.2),
        # preprocessing.Rescaling(1./255)  # use for models like ResNet50V2, but not EfficientNet
    ], name="data_augmentation")

    return data_augmentation


def visualize_random_img(data_augmentation, train_data):
    # visualize data augmentation
    # view a random image and compare to augmented version
    target_class = random.choice(train_data.class_names)
    target_dir = os.path.join(du.TRAIN_DATA_PATH_1_PERCENT, target_class)

    # select a random image and plot it
    random_image = random.choice(os.listdir(target_dir))
    random_image_path = os.path.join(target_dir, random_image)
    img = mpimg.imread(random_image_path)

    _, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.suptitle(f"Random images from {target_class} class")
    plt.sca(axes[0])
    plt.imshow(img)
    plt.xlabel("Original image")
    axes[0].set_yticklabels([])
    axes[0].set_xticklabels([])
    axes[0].set_yticks([])
    axes[0].set_xticks([])

    # augment image and plot
    augmented_img = data_augmentation(tf.expand_dims(img, axis=0))
    plt.sca(axes[1])
    plt.imshow(tf.squeeze(augmented_img) / 255.)
    plt.xlabel("Augmented image")
    axes[1].set_yticklabels([])
    axes[1].set_xticklabels([])
    axes[1].set_yticks([])
    axes[1].set_xticks([])
    plt.show()


def experiment_one(data_augmentation, train_data, test_data):
    """
    feature extraction transfer learning on 1% of the data with data augmentation
    :return:
    """

    # set up input shape & base model with base model layers frozen
    input_shape = (du.IMG_SIZE, du.IMG_SIZE, 3)
    base_model = tf.keras.applications.EfficientNetB0(include_top=False)
    base_model.trainable = False

    # Create input layer
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # Add data augmentation as a layer
    x = data_augmentation(inputs)

    # Give base_model inputs (after augmentation) and don't train it
    x = base_model(x, training=False)

    # pool output features of the base model
    x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

    # put dense layer on as output
    outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)

    # make a model using inputs and outputs
    model = keras.Model(inputs, outputs)

    # compile the model
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

    # fit the model
    history = model.fit(train_data, epochs=5, steps_per_epoch=len(train_data),
                        validation_data=test_data, validation_steps=int(0.25 * len(test_data)),
                        callbacks=[create_tensorboard_callback(dir_name="transfer_learning",
                                                               experiment_name="1_percent_data_aug")])

    # results = model.evaluate(test_data)

    plot_loss_curves(history)


def experiment_two(train_data, test_data, plot_curves=False):
    # create data augmentation layers
    data_augmentation = keras.Sequential([
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomHeight(0.2),
        preprocessing.RandomWidth(0.2),
        preprocessing.RandomZoom(0.2),
        preprocessing.RandomRotation(0.2),
    ], name="data_augmentation")

    # set up the input shape of our model
    input_shape = (224, 224, 3)

    # Create a frozen base model (also called the backbone)
    base_model = keras.applications.EfficientNetB0(include_top=False)
    base_model.trainable = False

    # Create the inputs and outputs and layers in between
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = data_augmentation(inputs)  # augment our training images (does not affect test data)
    x = base_model(x,
                   training=False)  # pass augmented images to the base model but keep it in inference mode, this also insures that batchnorm layers don't get updated, see: https://keras.io/guides/transfer_learning/#build-a-model
    x = layers.GlobalAveragePooling2D(name="global_avg_pooling_2d")(x)
    outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)

    model = keras.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(),
                  metrics=["accuracy"])

    # print(model.summary())
    # add ModelCceckpoint callback to save model during training
    checkpoint_path = "checkpoints/ten_pct_mod_wts/checkpoint.ckpt"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                          save_weights_only=True,
                                                          save_best_only=True,
                                                          save_freq="epoch",
                                                          verbose=1)

    initial_epochs = 5
    history = model.fit(train_data,
                        epochs=initial_epochs,
                        validation_data=train_data,
                        validation_steps=int(0.25 * len(train_data)),
                        callbacks=[create_tensorboard_callback(dir_name="transfer_learning",
                                                               experiment_name="10_percent_data_aug"),
                                   checkpoint_callback],
                        workers=-1)
    results_10_percent_data = model.evaluate(test_data)
    if plot_curves:
        plot_loss_curves(history)

    # load saved weights from checkpoint
    # this returns a model to a specific checkpoint
    # # model.load_weights(checkpoint_path)

    # results_loaded_weights = model.evaluate(test_data)

    # results should be the same, but there is precision difference
    # so use np.isclose() to compare
    # print(results_10_percent_data == results_loaded_weights)
    # print(np.isclose(np.array(results_10_percent_data), np.array(results_loaded_weights)))
    # print(np.array(results_10_percent_data) - np.array(results_loaded_weights))

    return model


def experiment_three(model, test_data, train_data):
    """
    efficient net with fine tuning, with some layers unfrozen for training
    Fine tuning usually works best _after_ training a feature extraction model for a few epochs with large amounts of
    custom data. So train the model for some epochs, and then unfreeze the layers for fine tuning and more training
    :return:
    """
    # visualize layer accessibility:
    # for layer in model.layers:
    #     print(layer, layer.trainable)

    # look closer at base model layer set to not trainable:#
    # for i, layer in enumerate(model.layers[2].layers):
    #     print(i, layer.name, layer.trainable)

    model.trainable = True
    # now freeze all layers except the last 10
    for layer in model.layers[2].layers[:-10]:
        layer.trainable = False

    # Must recompile model every time it is changed
    # lower the learning rate by factor of 10 when fine tuning to reduce (see ULMFit paper)
    model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=0.0001), metrics=["accuracy"])

    # now check trainable layers again
    for i, layer in enumerate(model.layers[2].layers):
        print(i, layer.name, layer.trainable)


def run():
    # walk_through_dir(du.LOCAL_DATA_PATH_1_PERCENT)

    # load datasets from files
    train_data_1_percent = tf.keras.preprocessing.image_dataset_from_directory(du.TRAIN_DATA_PATH_1_PERCENT,
                                                                               label_mode="categorical",
                                                                               image_size=du.IMG_SHAPE,
                                                                               batch_size=du.BATCH_SIZE)

    test_data = tf.keras.preprocessing.image_dataset_from_directory(du.TEST_DATA_PATH_1_PERCENT,
                                                                    label_mode="categorical",
                                                                    image_size=du.IMG_SHAPE,
                                                                    batch_size=du.BATCH_SIZE)

    # data_augmentation = set_up_data_aug()
    # visualize_random_img(data_augmentation, train_data_1_percent)
    # experiment_one(data_augmentation, train_data_1_percent, test_data)

    train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(du.TRAIN_DATA_PATH,
                                                                                label_mode="categorical",
                                                                                image_size=du.IMG_SHAPE)
    test_data = tf.keras.preprocessing.image_dataset_from_directory(du.TEST_DATA_PATH,
                                                                    label_mode="categorical",
                                                                    image_size=du.IMG_SHAPE)

    model = experiment_two(train_data_10_percent, test_data)
    experiment_three(model, train_data_10_percent, test_data)

