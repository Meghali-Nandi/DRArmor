import yaml
import flwr as fl
import numpy as np
import tensorflow as tf

from utils.dataset_loader import load_dataset
from clients.flower_client import FlowerClient
from models.mnist_model import get_mnist_model
from models.cifar10_model import get_cifar10_model
from models.cifar100_model import get_cifar100_model
from models.imagenet_model import get_imagenet_model
from models.resnet18 import get_resnet18

def main():
    # 1) Load YAML config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 2) Unpack federated settings
    dataset_name   = cfg["dataset"]["name"]
    num_clients    = cfg["federated"]["num_clients"]
    global_rounds  = cfg["federated"]["global_rounds"]
    local_epochs   = cfg["federated"]["local_epochs"]
    batch_size     = cfg["federated"]["batch_size"]

    # 3) Load and shard dataset
    train_shards, test_ds, num_classes = load_dataset(
        dataset_name,
        batch_size=batch_size,
        num_clients=num_clients
    )

    # 4) Map dataset to model function
    model_map = {
        "mnist":       get_mnist_model,
        "cifar10":     get_cifar10_model,
        "cifar100":    get_cifar100_model,
        "imagenet":    get_imagenet_model,
        "cats_v_dogs": get_resnet18,
    }
    model_fn = model_map[dataset_name.lower()]

    # 5) Flower client factory
    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        # Each FlowerClient wraps the DRArmor-equipped FLClient
        return FlowerClient(
            client_id          = idx,
            model_fn           = model_fn,
            train_dataset      = train_shards[idx],
            test_dataset       = test_ds,
            num_classes        = num_classes,
            config             = cfg,
        )

    # 6) Choose FedAvg or FedSGD strategy
    strategy_cfg = cfg["aggregation"]
    if strategy_cfg["algorithm"].lower() == "fedavg":
        strategy = fl.server.strategy.FedAvg(
            fraction_fit       = strategy_cfg.get("fraction_fit", 1.0),
            fraction_evaluate  = strategy_cfg.get("fraction_eval", 1.0),
            min_fit_clients    = num_clients,
            min_evaluate_clients = num_clients,
            min_available_clients = num_clients,
        )
    elif strategy_cfg["algorithm"].lower() == "fedsgd":
        strategy = fl.server.strategy.FedSGD(
            fraction_fit       = strategy_cfg.get("fraction_fit", 1.0),
            min_fit_clients    = num_clients,
            min_available_clients = num_clients,
        )
    else:
        raise ValueError(f"Unknown aggregation algorithm: {strategy_cfg['algorithm']}")

    # 7) Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=global_rounds),
        strategy=strategy,
        client_manager=fl.server.client_manager.SimpleClientManager(),
        client_selection=fl.server.client_selection.AllClientSelection(),  # all clients each round
        # call client_fn for each client
        client_fn=client_fn
    )

if __name__ == "__main__":
    main()
