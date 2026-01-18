"""
Training script for Quantum Splitting CNN Disease Detection Model.

This script implements training for the ISIC 2017 skin cancer classification task.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backbone_3 import QCNet
from data import get_dataloaders, ISIC2017Dataset
from utils import (
    set_seed,
    get_device,
    count_parameters,
    save_checkpoint,
    EarlyStopping,
    AverageMeter,
    setup_logging,
    compute_metrics,
)
from config import TRAINING_CONFIG, MODEL_CONFIG, RANDOM_SEED


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger=None,
) -> Tuple[float, float]:
    """
    Train model for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to use.
        epoch: Current epoch number.
        logger: Optional logger.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        accuracy = correct / labels.size(0)

        # Update meters
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(accuracy, labels.size(0))

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"})

    return loss_meter.avg, acc_meter.avg


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate model on validation set.

    Args:
        model: Model to validate.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Device to use.
        epoch: Current epoch number.

    Returns:
        Tuple of (average_loss, accuracy, all_labels, all_preds, all_probs).
    """
    model.eval()

    loss_meter = AverageMeter()
    all_labels = []
    all_preds = []
    all_probs = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            loss_meter.update(loss.item(), labels.size(0))

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy = (all_labels == all_preds).mean()

    return loss_meter.avg, accuracy, all_labels, all_preds, all_probs


def train(
    data_root: str,
    output_dir: str = "outputs",
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 15,
    resume: Optional[str] = None,
    seed: int = 42,
) -> None:
    """
    Main training function.

    Args:
        data_root: Path to ISIC 2017 dataset.
        output_dir: Directory to save outputs.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for optimizer.
        patience: Early stopping patience.
        resume: Path to checkpoint to resume from.
        seed: Random seed.
    """
    # Set up
    set_seed(seed)
    device = get_device()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    logger = setup_logging(str(output_path / "train.log"))
    logger.info(f"Training on device: {device}")

    # Create data loaders
    logger.info("Loading dataset...")
    train_loader, val_loader, _ = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        image_size=(224, 224),
        num_workers=4,
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    # Log class distribution for debugging
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    if hasattr(train_dataset, "get_class_distribution"):
        train_dist = train_dataset.get_class_distribution()
        val_dist = val_dataset.get_class_distribution()
        logger.info(f"Train class distribution: {train_dist}")
        logger.info(f"Val class distribution: {val_dist}")

    # Create model
    logger.info("Creating model...")
    model = QCNet()
    model = model.to(device)

    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Get class weights for imbalanced data
    train_dataset = train_loader.dataset
    if hasattr(train_dataset, "get_class_weights"):
        class_weights = train_dataset.get_class_weights().to(device)
        logger.info(f"Class weights: {class_weights}")
    else:
        class_weights = None

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode="max")

    # Resume from checkpoint
    start_epoch = 0
    best_accuracy = 0.0

    if resume is not None and os.path.exists(resume):
        logger.info(f"Resuming from checkpoint: {resume}")
        from utils import load_checkpoint

        start_epoch, _, best_accuracy = load_checkpoint(
            resume, model, optimizer, scheduler, device
        )
        start_epoch += 1

    # Training loop
    logger.info("Starting training...")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )

        # Validate
        val_loss, val_acc, val_labels, val_preds, val_probs = validate(
            model, val_loader, criterion, device, epoch
        )

        # Update scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log metrics
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, "
            f"LR={scheduler.get_last_lr()[0]:.6f}, "
            f"Time={epoch_time:.1f}s"
        )

        # Warn if accuracy is suspiciously high (potential data leakage)
        if train_acc >= 0.99 and val_acc >= 0.99 and epoch < 5:
            logger.warning(
                "WARNING: Both train and validation accuracy are >= 99% in early epochs. "
                "This may indicate data leakage (overlapping images between train/val sets). "
                "Please verify your dataset configuration."
            )

        # Compute detailed metrics
        metrics = compute_metrics(
            val_labels,
            val_preds,
            val_probs,
            num_classes=3,
            class_names=["melanoma", "seborrheic_keratosis", "benign"],
        )

        logger.info(
            f"  F1 Macro={metrics['f1_macro']:.4f}, "
            f"Precision={metrics['precision_macro']:.4f}, "
            f"Recall={metrics['recall_macro']:.4f}"
        )

        # Save checkpoint
        is_best = val_acc > best_accuracy
        if is_best:
            best_accuracy = val_acc
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                val_acc,
                str(checkpoint_dir / "best_model.pth"),
                scheduler,
                {"metrics": metrics},
            )
            logger.info(f"  New best model saved! Accuracy: {best_accuracy:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                val_acc,
                str(checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"),
                scheduler,
            )

        # Early stopping
        if early_stopping(val_acc):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    logger.info(f"Training completed. Best accuracy: {best_accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Quantum Disease Detection Model"
    )

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to ISIC 2017 dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAINING_CONFIG["batch_size"],
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_CONFIG["epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=TRAINING_CONFIG["learning_rate"],
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=TRAINING_CONFIG["weight_decay"],
        help="Weight decay",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=TRAINING_CONFIG["early_stopping_patience"],
        help="Early stopping patience",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed",
    )

    args = parser.parse_args()

    train(
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        resume=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
