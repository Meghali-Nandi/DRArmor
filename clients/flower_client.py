import flwr as fl
import numpy as np
import tensorflow as tf

from .client import FLClient  # your existing class

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, model_fn, train_ds, test_ds, config):
        # Wrap your FLClient
        self.inner = FLClient(client_id, model_fn, (train_ds, test_ds, None), config)

    def get_parameters(self):
        # Return model weights as a list of NumPy arrays
        return self.inner.model.get_weights()

    def fit(self, parameters, config):
        # Set the global model weights
        self.inner.model.set_weights(parameters)
        # Perform local training (one round)
        new_weights = self.inner.local_update(epochs=config["local_epochs"])
        return new_weights, len(self.inner.train_ds.dataset)

    def evaluate(self, parameters, config):
        # Set model and evaluate on local test set
        self.inner.model.set_weights(parameters)
        loss = tf.keras.metrics.Mean()
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        for x, y in self.inner.test_ds:
            preds = self.inner.model(x, training=False)
            loss.update_state(tf.keras.losses.sparse_categorical_crossentropy(y, preds))
            accuracy.update_state(y, preds)
        return float(loss.result()), len(self.inner.test_ds), {"accuracy": float(accuracy.result())}
