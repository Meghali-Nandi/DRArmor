import tensorflow as tf
from tensorflow.keras import layers, models

def basic_block(x, filters, stride=1):
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

def get_resnet18():
    inputs = layers.Input((224,224,3))
    x = layers.Conv2D(64,7,strides=2,padding='same',use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(3,2,padding='same')(x)
    for filters, blocks in zip([64,128,256,512],[2,2,2,2]):
        for i in range(blocks): x = basic_block(x, filters, stride=2 if i==0 and filters>64 else 1)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000, activation=tf.nn.log_softmax)(x)
    return models.Model(inputs, x)
