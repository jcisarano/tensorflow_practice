import tensorflow as tf
import datetime


def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback instane to store log files.

    Stores log files with the filepath:
      "dir_name/experiment_name/current_datetime/"

    Args:
      dir_name: target directory to store TensorBoard log files
      experiment_name: name of experiment directory (e.g. efficientnet_model_1)
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def create_checkpoint_callback(checkpoint_path):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             save_freq="epoch",
                                                             verbose=1)
    return checkpoint_callback


def create_early_stopping_callback():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=1,
        restore_best_weights=True
    )
    return early_stopping


def create_reduce_lr_callback():
    reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,  # multiply by 0.2 (reduce by 5x)
        patience=2,
        verbose=1,
        min_lr=1e-7
    )
    return reduce_learning_rate


class MyEarlyStopping(tf.keras.callbacks.Callback):
    """Custom callback for stopping on reaching accuracy threshold
    Didn't test this -- it might work?
    For more complete example, see: https://www.tensorflow.org/guide/keras/custom_callback

    my_early_stop = MyEarlyStopping(0.95)
    """
    def __init__(self, acc_thresh):
        self.accuracy_threshold = acc_thresh

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > self.accuracy_threshold:
            self.model.stop_training = True
