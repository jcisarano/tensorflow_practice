import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

# paths to pretrained models
EFFICIENTNET_URL: str = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
RESNET_URL: str = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
MOBILENET_URL: str = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
MOBILENET_V2: str = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
INCEPTION_V3: str = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

LOCAL_DATA_PATH: str = os.path.join("datasets", "images/10_food_classes_10_percent")
TRAIN_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "train")
TEST_DATA_PATH: str = os.path.join(LOCAL_DATA_PATH, "test")

IMAGE_SHAPE = (224, 224)


def load_and_prep_data(train_dir=TRAIN_DATA_PATH, test_dir=TEST_DATA_PATH, batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1 / 255.)
    test_datagen = ImageDataGenerator(rescale=1 / 255.)
    train_data = train_datagen.flow_from_directory(train_dir,
                                                   batch_size=batch_size,
                                                   target_size=IMAGE_SHAPE,
                                                   class_mode="categorical")
    test_data = train_datagen.flow_from_directory(test_dir,
                                                  batch_size=batch_size,
                                                  target_size=IMAGE_SHAPE,
                                                  class_mode="categorical")

    return train_data, test_data


def create_simple_model(model_path, num_classes: int = 10, input_shape=IMAGE_SHAPE):
    model = tf.keras.models.Sequential([
        hub.KerasLayer(model_path, input_shape=input_shape + (3,), trainable=False, name="feature_extractor_layer"),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    print(model.summary())

    return model


def create_model_frm_tf(num_classes: int = 10, input_shape=IMAGE_SHAPE):
    # base_model = tf.keras.applications.EfficientNetB0(include_top=False)
    base_model = tf.keras.applications.ResNet50(include_top=False)
    base_model.trainable = False
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pooling_2d")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

    print(model.summary())

    return model


def create_model_w_unlocked_layers(num_classes: int = 10, input_shape=IMAGE_SHAPE):
    model = create_model_frm_tf(num_classes, input_shape)
    model.trainable = True
    for layer in model.layers[:-5]:
        layer.trainable = False

    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=["accuracy"])
    print(model.summary())
    return model


def run():
    train_data, test_data = load_and_prep_data()
    # model = create_simple_model(RESNET_URL)
    # model = create_model_frm_tf(input_shape=(224, 224, 3))
    model = create_model_w_unlocked_layers(input_shape=(224, 224, 3))

    model.fit(train_data,
              epochs=50,
              steps_per_epoch=len(train_data),
              validation_data=test_data,
              validation_steps=int(0.25 * len(test_data)),
              workers=-1)

    print("transfer learning")
