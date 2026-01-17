"""
Utility functions for training and evaluation.
"""

import os
import random
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the best available device.

    Returns:
        torch.device for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    path: str,
    scheduler: Optional[_LRScheduler] = None,
    additional_info: Optional[Dict] = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch.
        loss: Current loss.
        accuracy: Current accuracy.
        path: Path to save checkpoint.
        scheduler: Optional learning rate scheduler.
        additional_info: Optional additional information to save.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if additional_info is not None:
        checkpoint.update(additional_info)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, float, float]:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to load state into.
        scheduler: Optional scheduler to load state into.
        device: Device to load checkpoint to.

    Returns:
        Tuple of (epoch, loss, accuracy).
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return (
        checkpoint.get("epoch", 0),
        checkpoint.get("loss", 0.0),
        checkpoint.get("accuracy", 0.0),
    )


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' for loss, 'max' for accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric.

        Returns:
            True if training should stop.
        """
        if self.mode == "min":
            score = -score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update statistics.

        Args:
            val: Current value.
            n: Number of samples.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_file: Optional path to log file.
        level: Logging level.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger("quantum_disease_detection")
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
