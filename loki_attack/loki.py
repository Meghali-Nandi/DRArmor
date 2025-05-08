import os, sys, subprocess

# As the Loki repository is licensed, we cannot upload the working code of Loki here.
# Please clone the repo from https://github.com/Manishpandey-0/Adversarial-reconstruction-attack-on-FL-using-LOKI
# and use the repo to return the ReconstructionAttack

try:
    from attack import ReconstructionAttack
except ImportError:
    class ReconstructionAttack:
        def __init__(self, model):
            self.model = model
        def run(self, data):
            raise NotImplementedError("Ensure LoKI repo structure matches import path.")

class LokiAttack:
    def __init__(self, model):
        self.attack = ReconstructionAttack(model)
    def extract_gradients(self, data):
        return self.attack.run(data)