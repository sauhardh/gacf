import numpy as np
from sklearn.metrics import roc_auc_score


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)  # highest softmax score per sample
    predictions = probs.argmax(axis=1)  # predicted class
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if in_bin.sum() == 0:
            continue
        avg_conf = confidences[in_bin].mean()
        avg_acc = correct[in_bin].mean()
        ece_val += in_bin.sum() * abs(avg_conf - avg_acc)

    return ece_val / len(labels)


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """
    Computes all validation metrics in one call.
    Returns a dict so adding new metrics later doesn't break any callers.
    """
    auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    return {
        "auc": auc,
        "ece": ece(probs, labels),
    }
