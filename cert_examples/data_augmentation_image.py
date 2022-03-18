from keras_preprocessing.image import ImageDataGenerator


def load_minibatch_data_augmented(train_dir, test_dir, img_size, shuffle_data=True):
    train_datagen_augmented = ImageDataGenerator(rescale=1 / 255.,
                                                 rotation_range=0.2,  # how much to rotate image
                                                 shear_range=0.2,  #
                                                 zoom_range=0.2,  # how much to enlarge/shrink
                                                 width_shift_range=0.2,  # left/right movement
                                                 height_shift_range=0.2,  # up/down movement
                                                 horizontal_flip=True)

    train_datagen = ImageDataGenerator(rescale=1 / 255.)
    test_datagen = ImageDataGenerator(rescale=1 / 255.)

    train_data_augmented = train_datagen_augmented.flow_from_directory(directory=train_dir,
                                                                       target_size=img_size,
                                                                       class_mode="binary",
                                                                       batch_size=32,
                                                                       shuffle=shuffle_data)  # for training purposes only, usually shuffle is good

    train_data = train_datagen.flow_from_directory(directory=train_dir,
                                                   target_size=img_size,
                                                   class_mode="binary",
                                                   batch_size=32,
                                                   shuffle=shuffle_data)  # for training purposes only, usually shuffle is good
    test_data = test_datagen.flow_from_directory(directory=test_dir,
                                                 target_size=img_size,
                                                 class_mode="binary",
                                                 batch_size=32,
                                                 shuffle=shuffle_data)  # for training purposes only, usually shuffle is good

    return train_data, train_data_augmented, test_data