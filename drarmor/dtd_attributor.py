import tensorflow as tf
import numpy as np
from scipy.stats import wasserstein_distance

class DTDAttributor:
    def __init__(self): pass

    def compute_relevance(self, model, gradients, x_batch, baseline):
        # Deep Taylor Decomposition: relevance = grad * (x - x0)
        relevance = {}
        diff = tf.cast(x_batch - baseline, tf.float32)
        for w, grad in zip(model.trainable_weights, gradients):
            rel = tf.reduce_sum(tf.abs(grad * diff)) if isinstance(diff, tf.Tensor) else tf.reduce_sum(tf.abs(grad))
            relevance[w.name] = float(rel.numpy())
        return relevance

    def identify_malicious(self, relevance, threshold=None):
        scores = np.array(list(relevance.values()))
        # threshold by Wasserstein distance from zero
        dist = wasserstein_distance(scores, np.zeros_like(scores))
        return [name for name, score in relevance.items() if score > dist]