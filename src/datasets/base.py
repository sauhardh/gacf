from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import pandas as pd


class BaseDatasetAdapter(ABC):
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """How many output classes this dataset has"""
        ...

    @property
    @abstractmethod
    def class_names(self) -> list:
        """Human-readable class label strings."""
        ...

    @abstractmethod
    def get_fold(self, fold_idx: int, split: str, transform) -> Dataset:
        """
        Returns a PyTorch Dataset for the given fold and split.

        Args:
        fold_idx: which fold (0 to num_fold-1)
        split: 'train' or 'val'
        transform: torchvision transform to apply
        """
        ...
