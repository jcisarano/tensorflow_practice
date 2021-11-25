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
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

import data_utils
from helper_functions import plot_loss_curves


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

