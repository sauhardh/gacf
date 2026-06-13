"""
CIFAR10 (official train + official test)
                ↓
           Merge all data
                ↓
      StratifiedKFold(n_splits=5)
                ↓
 Fold 0: Train(80%) + Val(20%)
 Fold 1: Train(80%) + Val(20%)
 Fold 2: Train(80%) + Val(20%)
 Fold 3: Train(80%) + Val(20%)
 Fold 4: Train(80%) + Val(20%)
"""

from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np

from src.pre.datasets.base import BaseDatasetAdapter


class CIFAR10Adapter(BaseDatasetAdapter):
    """
    Fold strategy: we combine train+test splits into one pool,
    then apply StratifiedKFold so every fold has the same
    class distribution.
    """

    CLASS_NAMES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def __init__(self, data_dir: str, num_folds: int = 5, seed: int = 42):
        train_ds = CIFAR10(root=data_dir, train=True, download=True)
        test_ds = CIFAR10(root=data_dir, train=False, download=True)

        all_data = np.concatenate([train_ds.data, test_ds.data], axis=0)
        all_labels = train_ds.targets + test_ds.targets

        self._data = all_data
        self._labels = np.array(all_labels)

        self._fold_indices = self._make_folds(num_folds, seed)

    def _make_folds(self, n_splits, seed):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(self._data, self._labels))

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def class_names(self):
        return self.CLASS_NAMES

    def get_fold(self, fold_idx: int, split: str, transform) -> Dataset:
        train_idx, val_idx = self._fold_indices[fold_idx]
        indices = train_idx if split == "train" else val_idx

        return _RawArrayDataset(self._data[indices], self._labels[indices], transform)


class _RawArrayDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image

        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])
