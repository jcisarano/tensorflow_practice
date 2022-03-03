import tensorflow as tf
import numpy as np

from custom_loss import load_and_prep_data, create_huber


class HuberMetric(tf.keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold":self.threshold}


def model_w_custom_class(X_train_scaled, y_train, input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=[HuberMetric(2.0)])
    model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2, workers=-1)

    save_path = "saved_models/model_w_custom_metric_class.h5"
    model.save(save_path)
    loaded_model = tf.keras.models.load_model(save_path,
                                              custom_objects={
                                                  "huber_fn": create_huber(2.0),
                                                  "HuberMetric": HuberMetric
                                              })
    loaded_model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2)
    print(loaded_model.metrics[-1].threshold)


def simple_model_w_custom_metric(X_train_scaled, y_train, input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal", input_shape=input_shape),
        tf.keras.layers.Dense(1),
    ])
    model.compile(loss="mse", optimizer="nadam", metrics=[create_huber(2.0)])
    model.fit(X_train_scaled, y_train, epochs=2, workers=-1)

    # show differenc between loss and metric due to floating point precision errors
    model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=[create_huber(2.0)])
    sample_weight = np.random.rand(len(y_train))
    history = model.fit(X_train_scaled, y_train, epochs=2, sample_weight=sample_weight)
    print(history.history["loss"][0], history.history["huber_fn"][0] * sample_weight.mean())


def run():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_and_prep_data()
    input_shape = X_train_scaled.shape[1:]

    tf.keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    # simple_model_w_custom_metric(X_train_scaled, y_train, input_shape)
    model_w_custom_class(X_train_scaled, y_train, input_shape)



    print("custom metrics")