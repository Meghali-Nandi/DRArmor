import tensorflow as tf
from tensorflow.keras import layers, models

def residual_block(x, filters, stride=1):
    shortcut = x
    x = layers.Conv2D(filters,3,strides=stride,padding='same',use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters,3,strides=1,padding='same',use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if stride!=1 or shortcut.shape[-1]!=filters:
        shortcut = layers.Conv2D(filters,1,strides=stride,use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

def get_cifar100_model():
    inputs = layers.Input(shape=(32,32,3))
    x = layers.Conv2D(64,3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    # Depth: 2 blocks per filter size
    for i in range(2): x = residual_block(x,64,1)
    for i in range(2): x = residual_block(x,128,2 if i==0 else 1)
    for i in range(2): x = residual_block(x,256,2 if i==0 else 1)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(100)(x)
    outputs = layers.Activation(tf.nn.log_softmax)(x)
    return models.Model(inputs, outputs)