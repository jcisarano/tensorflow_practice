import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

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


def run():
    train_data, test_data = load_and_prep_data()
    print("transfer learning")
