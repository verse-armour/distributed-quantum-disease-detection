"""
Testing/Evaluation script for Quantum Splitting CNN Disease Detection Model.

This script evaluates the trained model on the test set and generates metrics.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backbone_3 import QCNet
from data import get_single_loader, ISIC2017Dataset
from utils import (
    set_seed,
    get_device,
    load_checkpoint,
    compute_metrics,
    compute_confusion_matrix,
    get_classification_report,
    compute_sensitivity_specificity,
    setup_logging,
)
from matrix import plot_matrix
from config import RANDOM_SEED


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Evaluate model on test set.

    Args:
        model: Trained model.
        test_loader: Test data loader.
        device: Device to use.

    Returns:
        Tuple of (all_labels, all_preds, all_probs).
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    pbar = tqdm(test_loader, desc="Evaluating", leave=True)

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def test(
    data_root: str,
    checkpoint_path: str,
    output_dir: str = "outputs",
    batch_size: int = 32,
    save_confusion_matrix: bool = True,
    seed: int = 42,
) -> dict:
    """
    Main test function.

    Args:
        data_root: Path to ISIC 2017 dataset.
        checkpoint_path: Path to trained model checkpoint.
        output_dir: Directory to save outputs.
        batch_size: Batch size for evaluation.
        save_confusion_matrix: Whether to save confusion matrix plot.
        seed: Random seed.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Set up
    set_seed(seed)
    device = get_device()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(output_path / "test.log"))
    logger.info(f"Evaluating on device: {device}")

    # Create test data loader
    logger.info("Loading test dataset...")
    test_loader = get_single_loader(
        data_root=data_root,
        split="test",
        batch_size=batch_size,
        image_size=(224, 224),
        num_workers=4,
    )

    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    logger.info("Creating model...")
    model = QCNet()
    model = model.to(device)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    epoch, loss, accuracy = load_checkpoint(checkpoint_path, model, device=device)
    logger.info(f"Checkpoint from epoch {epoch}, val_loss={loss:.4f}, val_acc={accuracy:.4f}")

    # Evaluate
    logger.info("Running evaluation...")
    all_labels, all_preds, all_probs = evaluate(model, test_loader, device)

    # Class names
    class_names = ["melanoma", "seborrheic_keratosis", "benign"]

    # Compute metrics
    metrics = compute_metrics(
        all_labels,
        all_preds,
        all_probs,
        num_classes=3,
        class_names=class_names,
    )

    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    logger.info(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    logger.info(f"Recall (Macro): {metrics['recall_macro']:.4f}")

    if "auc_macro" in metrics:
        logger.info(f"AUC (Macro): {metrics['auc_macro']:.4f}")

    # Per-class metrics
    logger.info("\n" + "-" * 50)
    logger.info("Per-Class Metrics:")
    for name in class_names:
        logger.info(
            f"  {name}: "
            f"Precision={metrics.get(f'precision_{name}', 0):.4f}, "
            f"Recall={metrics.get(f'recall_{name}', 0):.4f}, "
            f"F1={metrics.get(f'f1_{name}', 0):.4f}"
        )

    # Sensitivity and Specificity
    sensitivity, specificity = compute_sensitivity_specificity(all_labels, all_preds, 3)
    logger.info("\n" + "-" * 50)
    logger.info("Sensitivity (Recall) and Specificity:")
    for i, name in enumerate(class_names):
        logger.info(
            f"  {name}: Sensitivity={sensitivity[i]:.4f}, Specificity={specificity[i]:.4f}"
        )

    # Classification report
    report = get_classification_report(all_labels, all_preds, class_names)
    logger.info("\n" + "-" * 50)
    logger.info("Classification Report:")
    logger.info("\n" + report)

    # Save confusion matrix
    if save_confusion_matrix:
        cm_path = str(output_path / "confusion_matrix.png")
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            plot_matrix(
                all_labels,
                all_preds,
                labels_name=[0, 1, 2],
                title="Confusion Matrix",
                axis_labels=class_names,
                save_path=cm_path,
            )
            logger.info(f"\nConfusion matrix saved to: {cm_path}")
        except Exception as e:
            logger.warning(f"Could not save confusion matrix: {e}")

    # Save metrics to file
    metrics_path = output_path / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("Evaluation Metrics\n")
        f.write("=" * 50 + "\n")
        for key, value in sorted(metrics.items()):
            f.write(f"{key}: {value:.4f}\n")

    logger.info(f"\nMetrics saved to: {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Quantum Disease Detection Model"
    )

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to ISIC 2017 dataset directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--no-confusion-matrix",
        action="store_true",
        help="Skip saving confusion matrix",
    )

    args = parser.parse_args()

    test(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        save_confusion_matrix=not args.no_confusion_matrix,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
