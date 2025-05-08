import tensorflow as tf
from tensorflow.keras import layers, models

def get_mnist_model():
    model = models.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(32,3, activation='relu'),
        layers.Conv2D(64,3, activation='relu'),
        layers.Conv2D(256,3, activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10),
        layers.Activation(tf.nn.log_softmax)
    ])
    return model
