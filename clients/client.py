import tensorflow as tf
from drarmor.engine import DRArmorEngine

class FLClient:
    def __init__(self, client_id, model_fn, dataset, config):
        self.id = client_id
        self.model = model_fn()
        self.train_ds, self.test_ds, _ = dataset
        self.engine = DRArmorEngine(
            self.model,
            threshold=config['defense']['threshold'],
            method=config['defense']['attribution_method']
        )
        self.dp_cfg = config['defense']['dp']

    def local_update(self, epochs=1):
        opt = tf.keras.optimizers.SGD(learning_rate=0.01)
        baseline = None
        for epoch in range(epochs):
            for x,y in self.train_ds:
                with tf.GradientTape() as tape:
                    preds = self.model(x, training=True)
                    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, preds))
                grads = tape.gradient(loss, self.model.trainable_weights)
                # first batch: set baseline for DTD
                if baseline is None: baseline = x
                # detect + defend
                model, grads_def = self.engine.detect_and_defend(grads, x, baseline)
                opt.apply_gradients(zip(grads_def, model.trainable_weights))
        return self.model.get_weights()
