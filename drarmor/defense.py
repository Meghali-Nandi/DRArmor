import numpy as np
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

class DefenseModule:
    def __init__(self, threshold, method='prune', dp_clip=1.0, noise_multiplier=1.0):
        """
        threshold: relevance threshold to identify malicious layers
        method: 'prune', 'dp', or 'both'
        dp_clip: L2 norm clipping bound for DP noise
        noise_multiplier: Gaussian noise multiplier for DP
        """
        self.threshold = threshold
        self.method = method
        self.dp_clip = dp_clip
        self.noise_multiplier = noise_multiplier
        if self.method in ('dp', 'both'):
            # DP optimizer used only to standardize noise parameters (not used directly for updates here)
            self.dp_optimizer = DPKerasSGDOptimizer(
                l2_norm_clip=self.dp_clip,
                noise_multiplier=self.noise_multiplier,
                num_microbatches=None,
                learning_rate=0.0
            )

    def apply(self, model, gradients, malicious_layers):
        # Option 1: Pruning malicious layers
        if self.method in ('prune', 'both'):
            for layer in model.layers:
                if layer.name in malicious_layers:
                    layer.trainable = False
        # Option 2: Differential Privacy via Gaussian noise
        if self.method in ('dp', 'both'):
            noisy_grads = []
            for g in gradients:
                # Clip gradients
                clipped = np.clip(g, -self.dp_clip, self.dp_clip)
                # Add Gaussian noise
                noise = np.random.normal(0, self.noise_multiplier * self.dp_clip, size=clipped.shape)
                noisy_grads.append(clipped + noise)
            return model, noisy_grads
        # Default: pruning only
        return model, gradients