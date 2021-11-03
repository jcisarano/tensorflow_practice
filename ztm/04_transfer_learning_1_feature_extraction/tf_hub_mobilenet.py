# use https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5
# on imagenet dataset

import tensorflow as tf
import tensorflow_hub as hub
import data_utils as du

MOBILENET_URL: str = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"


def create_model(path_to_model: str, num_layers: int = 10):
    feature_extraction_layer = hub.KerasLayer(path_to_model,
                                              trainable=False,
                                              name="feature_extraction_layer",
                                              input_shape=du.IMG_SHAPE + (3,))
    model = tf.keras.models.Sequential([
        feature_extraction_layer,
        tf.keras.layers.Dense(num_layers, activation="softmax", name="output_layer")
    ])
    return model


def run():
    train_data, test_data = du.load_and_prep_data()

    model = create_model(path_to_model=MOBILENET_URL)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    print(model)
    history = model.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))
    du.plot_loss_curve(history)