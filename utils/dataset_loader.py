import tensorflow as tf
import numpy as np

def load_dataset(name, batch_size, num_clients=1):
    if name.lower()=='mnist':
        (x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train[...,None]/255.0; x_test = x_test[...,None]/255.0
        num_classes=10
    elif name.lower()=='cifar-10':
        (x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
        x_train/=255.0; x_test/=255.0; num_classes=10
    elif name.lower()=='cifar-100':
        (x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar100.load_data()
        x_train/=255.0; x_test/=255.0; num_classes=100
    else:
        raise ValueError('Dataset not implemented')
    # Shuffle & shard
    idx = np.random.permutation(len(x_train))
    shards = np.array_split(idx, num_clients)
    train_ds = []
    for shard in shards:
        ds = tf.data.Dataset.from_tensor_slices((x_train[shard], y_train[shard]))\
               .shuffle(10000).batch(batch_size)
        train_ds.append(ds)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_ds, test_ds, num_classes   