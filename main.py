import yaml
import numpy as np

from clients.client import FLClient
from utils.dataset_loader import load_dataset
from models.mnist_model      import get_mnist_model
from models.cifar10_model    import get_cifar10_model
from models.cifar100_model   import get_cifar100_model
from models.imagenet_model   import get_imagenet_model
from models.resnet18         import get_resnet18

class FederatedServer:
    def __init__(self, model_fn, clients, aggregation):
        self.global_model = model_fn()
        self.clients = clients
        self.aggregation = aggregation

    def aggregate(self, client_weights):
        # FedAvg or FedSGD both reduce to averaging weights here
        avg = [
            np.mean([cw[i] for cw in client_weights], axis=0)
            for i in range(len(client_weights[0]))
        ]
        self.global_model.set_weights(avg)

    def run(self, rounds, local_epochs):
        for _ in range(rounds):
            weights = []
            for c in self.clients:
                c.model.set_weights(self.global_model.get_weights())
                w = c.local_update(epochs=local_epochs)
                weights.append(w)
            self.aggregate(weights)
        return self.global_model

if __name__ == '__main__':
    # 1) Load YAML config
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    # 2) Extract federated settings
    fed_cfg    = cfg['federated']
    num_clients   = fed_cfg['num_clients']
    rounds        = fed_cfg['global_rounds']
    local_epochs  = fed_cfg['local_epochs']
    batch_size    = fed_cfg['batch_size']

    # 3) Aggregation
    agg_algo = cfg['aggregation']['algorithm']

    # 4) Defense parameters
    defense_method = cfg['defense']['attribution_method']
    threshold      = cfg['defense']['threshold']
    dp_cfg         = cfg['defense']['dp']

    # 5) Dataset + shards
    ds_name = cfg['dataset']['name']
    train_shards, test_ds, num_classes = load_dataset(
        ds_name, batch_size, num_clients
    )

    # 6) Model selection
    model_map = {
        'mnist':       get_mnist_model,
        'cifar10':     get_cifar10_model,
        'cifar100':    get_cifar100_model,
        'imagenet':    get_imagenet_model,
        'cats_v_dogs': get_resnet18,
    }
    model_fn = model_map[ds_name]

    # 7) Instantiate clients, each with its shard
    clients = []
    for i in range(num_clients):
        client_ds = (train_shards[i], test_ds, num_classes)
        clients.append(
            FLClient(
                client_id   = i,
                model_fn    = model_fn,
                dataset     = client_ds,
                defense     = defense_method,
                threshold   = threshold,
                dp_config   = dp_cfg
            )
        )

    # 8) Run federated training
    server = FederatedServer(model_fn, clients, aggregation=agg_algo)
    final_model = server.run(rounds=rounds, local_epochs=local_epochs)
