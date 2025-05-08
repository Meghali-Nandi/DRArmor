from loki_attack.loki import LokiAttack
from drarmor.lrp_attributor import LRPAttributor
from drarmor.dtd_attributor import DTDAttributor
from drarmor.defense import DefenseModule

class DRArmorEngine:
    def __init__(self, model, threshold=0.5, use_lrp=False):
        self.model = model
        self.threshold = threshold
        self.attributor = LRPAttributor() if use_lrp else DTDAttributor()
        self.defense = DefenseModule(threshold)
        self.loki = LokiAttack(self.model)

    def run_attack(self, data):
        # Launch the adversarial LOKI attack on model
        return self.loki.extract_gradients(data)

    def detect_and_defend(self, gradients):
        relevance = self.attributor.compute_relevance(self.model, gradients)
        malicious_layers = self.attributor.identify_malicious(relevance)
        return self.defense.apply(self.model, gradients, malicious_layers)

    def step(self, data):
        gradients = self.run_attack(data)
        return self.detect_and_defend(gradients)