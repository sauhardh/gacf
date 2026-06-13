from src.pre.datasets.cifar10 import CIFAR10Adapter

DATA_REGISTRY = {
    "cifar10": CIFAR10Adapter,
}


def get_dataset(name: str, data_dir: str, num_folds: int, seed: int):
    if name not in DATA_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}' Available: {list(DATA_REGISTRY.keys())}"
        )

    return DATA_REGISTRY[name](data_dir, num_folds, seed)
