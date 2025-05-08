import tensorflow as tf

class LRPAttributor:
    def __init__(self): pass

    def compute_relevance(self, model, gradients, x_batch, baseline=None):
        # Compute layer relevance using gradient * activation (approximate LRP)
        relevance = {}
        # Forward pass to get activations
        activations = [layer.output for layer in model.layers]
        # Use gradient tape to recompute gradients w.r.t. activations
        inp = model.input
        with tf.GradientTape() as tape:
            tape.watch(activations)
            preds = model(x_batch)
            # sum of correct class logit
            loss = tf.reduce_sum(preds)
        grads_act = tape.gradient(loss, activations)
        for layer, grad_act in zip(model.layers, grads_act):
            # relevance = sum(abs(activation * grad_activation))
            activation = layer.output
            rel = tf.reduce_sum(tf.abs(activation * grad_act))
            relevance[layer.name] = float(rel.numpy())
        return relevance

    def identify_malicious(self, relevance, threshold):
        return [name for name, score in relevance.items() if score > threshold]