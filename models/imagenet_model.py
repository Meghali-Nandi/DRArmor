import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def get_imagenet_model():
    base = ResNet50(include_top=False, weights=None, input_shape=(224,224,3))
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(1000)(x)
    outputs = layers.Activation(tf.nn.log_softmax)(x)
    return models.Model(base.input, outputs)