from .cifar100_model import get_cifar100_model
import tensorflow as tf

def get_cifar10_model():
    model = get_cifar100_model()
    # adjust final layer for 10 classes
    model.layers[-2] = tf.keras.layers.Dense(10)
    return model