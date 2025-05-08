# DRArmor - Nosy Layers, Noisy Fixes: Tackling DRAs in Federated Learning Systems using Explainable AI

DRArmor is a modular framework for detecting and mitigating data reconstruction attacks (DRAs) in federated learning (FL). Building on the LOKI attack, DRArmor integrates explainable-AI (XAI) techniques—Layer-wise Relevance Propagation (LRP) and Deep Taylor Decomposition (DTD)—along with a Wasserstein-distance check to identify malicious layers. Once detected, clients can either prune those layers or apply Differential-Privacy (DP) Gaussian noise to their gradients before uploading to the server. DRArmor supports both FedAvg and FedSGD aggregation, and can run under a native Python loop or the Flower FL framework.

---

## 🚀 Key Features

- **Attack Integration**  
  - Wraps the adversarial reconstruction attack from [LOKI](https://github.com/Manishpandey-0/Adversarial-reconstruction-attack-on-FL-using-LOKI)  
- **Explainable-AI Defenses**  
  - **LRP**: recursive relevance propagation  
  - **DTD**: gradient-based Taylor decomposition  
  - **Wasserstein distance**: statistical check on layer-wise relevance  
- **Defense Actions**  
  - **Pruning**: remove malicious layers  
  - **DP-Gaussian**: clip and add Gaussian noise to gradients  
- **Aggregation Modes**  
  - **FedAvg** (weight averaging)  
  - **FedSGD** (gradient averaging via DP-Keras-SGD)  
- **Framework Support**  
  - Native Python FL loop  
  - Flower integration (`FlowerClient`, standard strategies)  
- **Configurable** via `config.yaml`: datasets, nodes, aggregation, defense, models, logging  
- **Model Zoo**  
  - MNIST CNN, CIFAR-10/100 ResNets, ImageNet ResNet, Cats v Dogs ResNet-18  
- **Dataset Sharding**: split any dataset across _n_ clients  

---

## 📦 Repository Structure

```text
DRArmor/
├── clients/                 
│   ├── client.py            
│   └── flower_client.py     
├── drarmor/                 
│   ├── engine.py            
│   ├── lrp_attributor.py    
│   ├── dtd_attributor.py    
│   └── defense.py           
├── loki_attack/             
│   └── loki.py              
├── models/                  
│   ├── mnist_model.py  
│   ├── cifar10_model.py  
│   ├── cifar100_model.py  
│   ├── imagenet_model.py  
│   └── resnet18.py  
├── utils/                   
│   └── dataset_loader.py    
├── requirements.txt         
├── config.yaml              
├── main.py                  
├── flower_main.py           
└── README.md                
