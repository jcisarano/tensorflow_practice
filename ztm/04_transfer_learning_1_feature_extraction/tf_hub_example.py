# Create a model using layers stored in TensorFlow Hub (https://tfhub.dev/)
# https://www.tensorflow.org/hub


import data_utils as du
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers


# pre-trained models from tensorflow hub
EFFICIENTNET_URL: str = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
RESNET_URL: str = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"


def create_model(model_url: str, num_classes: int=10):
    """
    Create model from tensorflow hub url and create a Sequential model
    :param model_url: TensorFlow Hub feature extraction URL
    :param num_classes: Number of neurons in output layer. Should be equal to number of target classes.
    :return: Uncompiled Keras Sequential model with model_url as feature extractor layer and Dense output layer with num_classes output neurons.
    """
    # Download the pretrained model and save it as a Keras layer
    feature_extraction_layer = hub.KerasLayer(model_url,
                                              trainable=False,  # will not affect pre-trained patterns
                                              name="feature_extraction_layer",
                                              input_shape=du.IMG_SHAPE+(3,))  # match desired img input shape
    model = tf.keras.models.Sequential([
        feature_extraction_layer,
        layers.Dense(num_classes, activation="softmax", name="output_layer")
    ])

    return model


def run():
    du.list_filecount_in_dir(dir=du.LOCAL_DATA_PATH)
    train_data, test_data = du.load_and_prep_data()

    model_resnet = create_model(model_url=RESNET_URL, num_classes=train_data.num_classes)
    print(model_resnet.summary())
    model_resnet.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

    history_resnet = model_resnet.fit(train_data,
                                      epochs=5,
                                      steps_per_epoch=len(train_data),
                                      validation_data=test_data,
                                      validation_steps=len(test_data),
                                      workers=-1)


