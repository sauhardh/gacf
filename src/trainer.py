import torch
import torch.nn as nn
import numpy as np
import wandb
from pathlib import Path

from src.metrics import compute_metrics


class Trainer:
    def __init__(self, model, cfg: dict, device, run_name: str, gacf_controller=None):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.run_name = run_name
        self.gacf_controller = gacf_controller

        self.criterion = nn.CrossEntropyLoss()

        # AdamW = Adam with proper weight decay decoupling
        # Only pass parameters that actually require gradients
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )

        # GradScaler manages the loss scaling for mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg["mixed_precision"])

        # Cosine annealing smoothly decays LR to 0 over num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg["num_epochs"]
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _run_epoch(self, loader, train: bool):
        self.model.train() if train else self.model.eval()

        total_loss, correct, total = 0.0, 0, 0
        all_probs, all_labels = [], []

        # torch.no_grad() when evaluating skips building the computation graph
        # which saves memory and speeds up the val pass significantly
        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # autocast automatically casts ops to float16 where safe
                with torch.amp.autocast("cuda", enabled=self.cfg["mixed_precision"]):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                if train:
                    self.optimizer.zero_grad()
                    # scaler.scale multiplies loss by a scale factor to prevent
                    # float16 underflow during backward
                    self.scaler.scale(loss).backward()
                    
                    # [GACF] Zero out gradients for frozen channels before stepping
                    if self.gacf_controller is not None:
                        self.gacf_controller.zero_frozen_gradients()
                        
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                probs = torch.softmax(outputs.float(), dim=1).detach().cpu().numpy()
                probs = probs / probs.sum(axis=1, keepdims=True)
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())

                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += images.size(0)

        return (
            total_loss / total,
            correct / total,
            np.concatenate(all_probs),
            np.concatenate(all_labels),
        )

    def fit(self, train_loader, val_loader, fold: int) -> float:
        """
        Full training run for one fold.
        Returns the best val_loss achieved (used to aggregate across folds).
        """
        Path("checkpoints").mkdir(exist_ok=True)

        for epoch in range(self.cfg["num_epochs"]):
            if self.gacf_controller is not None:
                self.gacf_controller.step_epoch(epoch)
                
            train_loss, train_acc, _, _ = self._run_epoch(train_loader, train=True)
            val_loss, val_acc, probs, labels = self._run_epoch(val_loader, train=False)

            metrics = compute_metrics(probs, labels)
            self.scheduler.step()

            log = {
                "epoch": epoch,
                "fold": fold,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": metrics["auc"],
                "val_ece": metrics["ece"],
                "lr": self.scheduler.get_last_lr()[0],
            }
            wandb.log(log)

            print(
                f"  ep {epoch:03d} | "
                f"loss {val_loss:.4f} | "
                f"acc {val_acc:.3f} | "
                f"auc {metrics['auc']:.3f} | "
                f"ece {metrics['ece']:.4f}"
            )

            # Checkpoint on improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                ckpt_path = f"checkpoints/{self.run_name}_fold{fold}_best.pt"
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  ✓ saved checkpoint → {ckpt_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg["early_stopping_patience"]:
                    print(
                        f"  early stop (patience={self.cfg['early_stopping_patience']})"
                    )
                    break

        return self.best_val_loss
