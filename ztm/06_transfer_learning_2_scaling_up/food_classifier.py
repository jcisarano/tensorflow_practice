"""
Large model that will classify all 101 food types in Food101 with 10% of training data

Steps to create the model:
1) Create a ModelCheckpoint callback
2) Create a data augmentation layer to build data augmentation directly into the model
3) Build a headless (no top layers) functional EfficientNetB0 backbone model (we will create our oun output model)
4) Compile
5) Feature extract for 5 full passes (5 epochs on train set and validate on 15% of test data to save time)

"""
import os

import keras
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
import pandas as pd
import random

import data_utils
from helper_functions import plot_loss_curves, make_confusion_matrix

MODEL_PATH: str = "saved_models/06_101_food_class_10_percent_saved_big_dog_model"


def train_model(train_data, test_data):
    # Create checkpoint callback
    checkpoint_path = "checkpoints/101_classes_10_percent_data_model_checkpoint"
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
    outputs = layers.Dense(len(train_data.class_names),
                           activation="softmax",
                           name="output_layer")(x)

    model = keras.Model(inputs, outputs)
    print(model.summary())

    # Compile() and Fit()
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics="accuracy")
    history = model.fit(train_data,
                        epochs=5,  # only five for now to keep experiments faster
                        validation_data=test_data,
                        validation_steps=int(0.15 * len(test_data)),  # using only 15% speeds up the epochs
                        callbacks=[checkpoint_callback],
                        workers=-1)

    results = model.evaluate(test_data, workers=-1)

    # plot_loss_curves(history)

    # Unfreeze every layer except the last 5
    # then refreeze all but the last five
    backbone.trainable = True
    for layer in backbone.layers[:-5]:
        layer.trainable = False

    # lower the learning rate by 10x for fine tuning
    # if we do another pass with more trainable layers, that should lower the learning rate even more
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics="accuracy")

    # see the trainable layers in overall model
    # if any parts of a layer is trainable, this will show the whole thing trainable
    for layer in model.layers:
        print(layer.name, layer.trainable)

    # see if EfficientNet layers are trainable:
    for index, layer in enumerate(model.layers[2].layers):
        print(index, layer.name, layer.trainable)

    # now fit again, starting from where the last one left off
    history_fine_tune = model.fit(train_data,
                                  epochs=10,  # five more than previous attempt
                                  validation_data=test_data,
                                  validation_steps=int(0.15 * len(test_data)),
                                  initial_epoch=history.epoch[-1],
                                  workers=-1)

    results_fine_tune = model.evaluate(test_data, workers=-1)
    print(results_fine_tune)
    # plot_loss_curves(history_fine_tune)

    # save and load model
    model_path = "saved_models/101_food_classes_10_percent_saved"
    model.save(model_path)

    # load and evaluate saved model
    model_loaded = tf.keras.models.load_model(model_path)
    results_loaded = model_loaded.evaluate(test_data)
    print(results_loaded)


def plot_horizontal_graph(test_data, classification_report_dict):
    class_f1_scores = {}
    for k, v in classification_report_dict.items():
        if k == "accuracy":
            break
        else:
            # print(test_data.class_names[int(k)], v["f1-score"])
            class_f1_scores[test_data.class_names[int(k)]] = v["f1-score"]

    # convert to dataframe
    f1_scores = pd.DataFrame({"class_names": list(class_f1_scores.keys()),
                              "f1-score": list(class_f1_scores.values())}).sort_values("f1-score", ascending=False)
    # print(f1_scores[:10])

    fix, axes = plt.subplots(figsize=(12, 25))
    scores = axes.barh(range(len(f1_scores)), f1_scores["f1-score"].values)
    axes.set_yticks(range(len(f1_scores)))
    axes.set_yticklabels(f1_scores["class_names"])
    axes.set_xlabel("F1-score")
    axes.set_title("F1-score for food classes")
    axes.invert_yaxis()
    plt.show()


def load_saved_model(model_path):
    return tf.keras.models.load_model(model_path)


def evaluate_saved_model(test_data):
    model = tf.keras.models.load_model("saved_models/06_101_food_class_10_percent_saved_big_dog_model")
    print(model.summary())

    # results = model.evaluate(test_data)
    # print(results)

    # make predictions
    pred_probs = model.predict(test_data, verbose=1)
    # print(pred_probs[0], len(pred_probs[0]), sum(pred_probs[0]))

    # print(f"Number of prediction probabilities for sample 0: {len(pred_probs[0])}")
    # print(f"Predictions for sample 0:\n {pred_probs[0]}")
    # print(f"The class for the best predicted probability for sample 0: {pred_probs[0].argmax()}")
    # print(f"Predicted class for sample 0: {test_data.class_names[pred_probs[0].argmax()]}")

    # predictions for each data instance:
    pred_classes = pred_probs.argmax(axis=1)
    # print(pred_classes)

    # unravel test data BatchDataset
    y_labels = []
    for images, labels in test_data.unbatch():
        y_labels.append(labels.numpy().argmax())  # get index of predicted class from one-hot encoded array
    # print(y_labels[:10])

    # Evaluate model's predictions
    # accuracy = accuracy_score(y_labels, pred_classes)
    # print(accuracy)

    # make_confusion_matrix(y_true=y_labels, y_pred=pred_classes, classes=test_data.class_names, figsize=(100, 100),
    #                      text_size=20, savefig=True)

    classification_report_dict = classification_report(y_true=y_labels,
                                                       y_pred=pred_classes,
                                                       output_dict=True)

    # plot_horizontal_graph(test_data, classification_report_dict)

    # now predict on a single image from the test set
    # ?

    find_most_wrong_predictions(test_data, y_labels, pred_classes, pred_probs)


def load_and_preprocess_image(filename, image_shape=224, normalize=True):
    """
    Reads an image filename, converts to tensor, and reshapes to (image_shape, image_shape, 3)
    Steps:
        Read a target image using tf.io.read_file()
        Turn image into Tensor using tf.io.decode_image()
        Resize image tensor to be the same size as training images using tf.image.resize()
        Scale image to get all pixel values between 0 & 1 if needed
    :param filename
    :param image_shape
    :param normalize
    :return:
    """
    img = tf.io.read_file(filename)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.resize(img, [image_shape, image_shape])
    if normalize:
        img = img / 255.

    return img


def predict_random_image(test_data, model):
    """
        Steps
            Load a few random images
            Make predictions
            Plot images along with predictions
    :return:
    """

    plt.figure(figsize=(16, 8))
    for i in range(3):
        class_name = random.choice(test_data.class_names)
        file_name = random.choice(os.listdir(os.path.join(data_utils.TEST_DATA_PATH, class_name)))
        file_path = os.path.join(data_utils.TEST_DATA_PATH, class_name, file_name)
        # print(file_path)
        img = load_and_preprocess_image(file_path, normalize=False)
        img_expanded = tf.expand_dims(img, axis=0)
        pred_prob = model.predict(img_expanded)
        pred_class = test_data.class_names[pred_prob.argmax()]
        # print(pred_prob, pred_class)
        # print(pred_class)

        plt.subplot(1, 3, i + 1)
        plt.imshow(img / 255.)
        if class_name == pred_class:
            title_color = "g"
        else:
            title_color = "r"

        plt.title(f"actual: {class_name}, pred: {pred_class}, prob: {pred_prob.max():.2f}", c=title_color)
        plt.axis(False)
    plt.show()


def find_most_wrong_predictions(test_data, y_labels, pred_classes, pred_probs):
    """
    'Most wrong' predictions are the wrong predictions with the highest probability. Identifying them can help you
    improve the model and improve the data.

    Steps:
        Get all of the image file paths in the test dataset using list_files()
        Create a pandas dataframe of image filepaths, ground truth labels, predicted classes, max pred probabilities
        Use dataframe to find all the wrong predictions
        Sort dataframe with wrong preds and highest probabilities at the top
        Visualize the images with the highest pred probabilities and wrong preds

    :return:
    """

    image_file_paths = []
    for file_path in test_data.list_files(os.path.join(data_utils.TEST_DATA_PATH, "*/*.jpg"), shuffle=False):
        # print(file_path.numpy())
        image_file_paths.append(file_path.numpy())

    pred_df = pd.DataFrame({"img_path": image_file_paths,
                            "y_true": y_labels,
                            "y_pred": pred_classes,
                            "pred_conf": pred_probs.max(axis=1),
                            "y_true_classname": [test_data.class_names[i] for i in y_labels],
                            "y_pred_classname": [test_data.class_names[i] for i in pred_classes]})

    pred_df["pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]
    top_100_wrong = pred_df[pred_df["pred_correct"] == False].sort_values("pred_conf", ascending=False)[:100]
    # print(top_100_wrong)

    start_index = 10
    images_to_view = 9
    plt.figure(figsize=(15, 15))
    for i, row in enumerate(top_100_wrong[start_index:start_index+images_to_view].itertuples()):
        plt.subplot(3, 3, i+1)
        print(row)
        _, path, _, _, pred_prob, y_true_class_name, y_pred_class_name, _ = row
        img = load_and_preprocess_image(path, normalize=True)
        plt.imshow(img)
        plt.title(f"actual: {y_true_class_name}, pred: {y_pred_class_name}\nprob: {pred_prob}", c="r")
        plt.axis(False)
    plt.show()


def run():
    train_data_all_10_percent \
        = tf.keras.preprocessing.image_dataset_from_directory(data_utils.TRAIN_DATA_PATH,
                                                              label_mode="categorical",
                                                              image_size=data_utils.IMG_SHAPE)
    test_data = tf.keras.preprocessing.image_dataset_from_directory(data_utils.TEST_DATA_PATH,
                                                                    label_mode="categorical",
                                                                    image_size=data_utils.IMG_SHAPE,
                                                                    shuffle=False)
    # train_model(train_data_all_10_percent, test_data)
    evaluate_saved_model(test_data)

    # model = load_saved_model(MODEL_PATH)
    # predict_random_image(test_data, model)
