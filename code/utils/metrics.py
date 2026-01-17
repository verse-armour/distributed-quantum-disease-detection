"""
Evaluation metrics for disease detection model.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 3,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Optional predicted probabilities for AUC.
        num_classes: Number of classes.
        class_names: Optional list of class names.

    Returns:
        Dictionary of metrics.
    """
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    for i, name in enumerate(class_names):
        if i < len(precision_per_class):
            metrics[f"precision_{name}"] = precision_per_class[i]
            metrics[f"recall_{name}"] = recall_per_class[i]
            metrics[f"f1_{name}"] = f1_per_class[i]

    # AUC if probabilities are provided
    if y_prob is not None:
        try:
            if num_classes == 2:
                metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics["auc_macro"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
                metrics["auc_weighted"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="weighted"
                )
        except ValueError:
            pass  # AUC undefined if not all classes present

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.
        normalize: Whether to normalize by row (true class).

    Returns:
        Confusion matrix array.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-10)

    return cm


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Generate classification report.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Optional list of class names.

    Returns:
        Classification report string.
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )


def compute_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 3,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute sensitivity (recall) and specificity per class.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.

    Returns:
        Tuple of (sensitivity_dict, specificity_dict).
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    sensitivity = {}
    specificity = {}

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return sensitivity, specificity
