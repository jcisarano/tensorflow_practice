"""
TensorFlow Datasets:
    ready-to-use datasets
    Already in tensor format
    Well-established datasets for many problem types
    Consistent
However, TF Datasets are static, they do not change like real-world datasets would
"""

"""
Neural networks perform best when data is formatted properly, e.g. batched, normalized, etc
However, raw data is rarely formatted as you will need, so you will need to write preprocessing functions.

What we know about our food data:
    unit8 datatype
    images are different sizes
    not normalized-- the color values range from 0 to 255
    
However, the models like data in the format:
    data in float32 type
    all tensors should be the same size
    tensors with normalized color values (0-1) usually work better
    
We'll use EfficientNetBX pretrained model, so we won't need to rescale data--it has scaling layer built in.
Our function will need to:
    1) reshape images
    2) change datatype from unit8 to float32
"""

"""
*** Very important resource on data import pipeline in TensorFlow: https://www.tensorflow.org/guide/data ***
"""

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from helper_functions import compare_histories, create_tensorboard_callback


def visualize_data(train_data, test_data, ds_info):
    """
    Explore the data to find:
        Class names
        Shape of input data (image tensors)
        Data type of input data
        What do the labels look like? One-hot encoding or label encoding?
        Do the labels match the class names?
    """
    print(ds_info.features["label"].names[:10])
    train_one_sample = train_data.take(1)
    plt.figure(figsize=(8, 5))
    for image, label in train_one_sample:
        print(f"""
        Image shape: {image.shape}
        Image datatype: {image.dtype}
        Target class from Food101 (tensor form): {label}
        Class name (str form): {ds_info.features["label"].names[label.numpy()]}
        """)
        # print(image)
        plt.subplot(121)
        plt.imshow(image)
        plt.title(f"class: {ds_info.features['label'].names[label.numpy()]}")
        plt.axis(False)
        print(f"Image before preprocessing:\n{image[:2]}...,\nShape: {image.shape},\nDataType: {image.dtype}\n")
        preprocessed_img = preprocess_img(image, label)
        print(f"Image after preprocessing:\n{preprocessed_img[:2]}...,\nShape: {preprocessed_img.shape},\n"
              f"DataType: {preprocessed_img.dtype}\n")
        plt.subplot(122)
        plt.imshow(preprocessed_img / 255.)
        plt.title(f"Preprocessed version")
        plt.axis(False)
        plt.show()


def preprocess_img(image, label, target_img_shape=224):
    """
    Converts img datatype to float32 and reshapes image to target_img_shape x target_img_shape X num_color_channels
    :param image:
    :param label:
    :param target_img_shape:
    :return:
    """
    image = tf.image.resize(image, [target_img_shape, target_img_shape])
    return tf.cast(image, tf.float32), label


def preprocess_datasets(train_data, test_data):
    # map preprocessing function to whole training data set (and parallelize)
    train_data_batched = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    # shuffle train_data and turn it into batches and prefetch it (to load faster)
    # 1000 is good value, but larger vals can be limited by available RAM
    # batching & prefetching with autotune makes the best possible use of all available CPU and GPU threads
    tratrain_data_batchedin_data = train_data_batched.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Map preprocessing function also for test_data
    test_data_batched = test_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    test_data_batched = test_data_batched.batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_data_batched, test_data_batched


def create_and_fit_model(ds_info):
    input_shape = (224, 224, 3)
    base_model = tf.keras.applications.EfficientNetB0(include_top=False)
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_layer")
    # Rescaling is not needed with EfficientNetB0 (it is built in)
    # x = preprocessing.Rescaling(1./255)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(len(ds_info.features["label"].names))(x)
    outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    print(model.summary())

    for layer in model.layers:
        print(layer.dtype)

    return model

def run():
    # a new way to load food101 dataset, from tensorflow_datasets
    datasets_list = tfds.list_builders()
    print("food101" in datasets_list)

    # load the full dataset, but it will take some time because the dataset is 5-6 gb
    # other datasets can be much larger, so check the dataset size first
    (train_data, test_data), ds_info = tfds.load(name="food101",
                                                 split=["train", "validation"],
                                                 shuffle_files=True,
                                                 as_supervised=True,  # includes labels in tuple (data,labels)
                                                 with_info=True  # includes meta data
                                                 )

    # visualize_data(train_data, test_data, ds_info)

    train_data, test_data = preprocess_datasets(train_data, test_data)

    print(train_data, test_data)

    # Create callbacks to help while training:
    #   Tensorboard callback to log training results (to visualize them later)
    #   ModelCheckpoint callback to save model progress after feature extraction
    # create_tensorboard_callback()
    checkpoint_path = "model_checkpoints/cp.ckpt"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                monitor="val_acc",
                                                save_best_only=True,
                                                save_weights_only=True,
                                                verbose=0)

    # Set up TensorFlow mixed precision training (see https://www.tensorflow.org/guide/mixed_precision)
    # mixed precision uses a combination of float32 and float16 data types to speed up processing
    mixed_precision.set_global_policy("mixed_float16")
    print(mixed_precision.global_policy())

    model = create_and_fit_model(ds_info)


