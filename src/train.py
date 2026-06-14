"""
Entry point for all baseline experiments.

Usage:
    python scripts/train.py --mode full_finetune
    python scripts/train.py --mode linear_probe  --dataset configs/datasets/isic.yaml
    python scripts/train.py --mode fixed_freeze  --config  configs/base.yaml
"""

import argparse
import random
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from src.config import load_config
from src.pre.datasets import get_dataset
from src.pre import get_transform
from src.model import build_model
from src.trainer import Trainer
from src.gacf import GACFController


def set_seed(seed: int):
    """Lock down every source of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(mode: str, cfg: dict):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | dataset: {cfg['dataset']['name']} | mode: {mode}")

    # Adapter handles all dataset-specific logic (folds, labels, paths)
    adapter = get_dataset(
        name=cfg["dataset"]["name"],
        data_dir=cfg["dataset"]["data_dir"],
        num_folds=cfg["num_folds"],
        seed=cfg["seed"],
    )

    # Override num_classes from the adapter (single source of truth)
    cfg["dataset"]["num_classes"] = adapter.num_classes

    fold_scores = []

    for fold in range(cfg["num_folds"]):
        print(f"\n{'=' * 50}")
        print(f"FOLD {fold} / {cfg['num_folds'] - 1}")
        print(f"{'=' * 50}")

        run_name = f"{cfg['dataset']['name']}_{mode}_fold{fold}"
        wandb.init(project="gacf-baselines", name=run_name, config=cfg, reinit=True)

        # adapter.get_fold returns a plain PyTorch Dataset — trainer doesn't
        # care what's inside it
        train_ds = adapter.get_fold(
            fold, "train", get_transform(cfg["img_size"], "train")
        )
        val_ds = adapter.get_fold(fold, "val", get_transform(cfg["img_size"], "val"))

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        model = build_model(cfg, mode=mode)
        
        gacf_controller = None
        if mode == "gacf":
            gacf_controller = GACFController(
                model=model,
                p_max=cfg.get("gacf_p_max", 0.5),
                t_freeze=cfg.get("gacf_t_freeze", 15),
                beta=cfg.get("gacf_beta", 0.9)
            )
            
        trainer = Trainer(model, cfg, device, run_name=run_name, gacf_controller=gacf_controller)
        score = trainer.fit(train_loader, val_loader, fold)

        fold_scores.append(score)
        wandb.finish()

    mean, std = np.mean(fold_scores), np.std(fold_scores)
    print(f"\n{'=' * 50}")
    print(f"{mode} complete | val_loss: {mean:.4f} ± {std:.4f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full_finetune", "linear_probe", "fixed_freeze", "gacf"],
        required=True,
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--dataset", default=None, help="Optional dataset override yaml"
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.dataset)
    run(args.mode, cfg)
