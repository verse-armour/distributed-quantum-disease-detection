"""
DataLoader utilities for ISIC 2017 dataset.

Provides functions to create train, validation, and test data loaders.
"""

import warnings
from pathlib import Path
from typing import Tuple, Optional, List, Set

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import ISIC2017Dataset


def validate_no_data_leakage(
    train_dataset: ISIC2017Dataset,
    val_dataset: ISIC2017Dataset,
    test_dataset: ISIC2017Dataset,
) -> bool:
    """
    Validate that there is no data leakage between train, validation, and test sets.
    
    This function checks for:
    1. Overlapping image paths between datasets
    2. Identical image directories being used for different splits
    
    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        
    Returns:
        True if no data leakage is detected, False otherwise.
        
    Raises:
        Warning if data leakage is detected.
    """
    has_leakage = False
    
    # Extract image paths from each dataset
    train_paths: Set[str] = {Path(p).name for p, _ in train_dataset.samples}
    val_paths: Set[str] = {Path(p).name for p, _ in val_dataset.samples}
    test_paths: Set[str] = {Path(p).name for p, _ in test_dataset.samples}
    
    # Check for overlapping image file names between splits
    train_val_overlap = train_paths & val_paths
    train_test_overlap = train_paths & test_paths
    val_test_overlap = val_paths & test_paths
    
    if train_val_overlap:
        warnings.warn(
            f"DATA LEAKAGE DETECTED: {len(train_val_overlap)} images appear in both "
            f"train and validation sets. This will cause inflated validation accuracy. "
            f"Example overlapping images: {list(train_val_overlap)[:3]}",
            UserWarning
        )
        has_leakage = True
        
    if train_test_overlap:
        warnings.warn(
            f"DATA LEAKAGE DETECTED: {len(train_test_overlap)} images appear in both "
            f"train and test sets. This will cause inflated test accuracy. "
            f"Example overlapping images: {list(train_test_overlap)[:3]}",
            UserWarning
        )
        has_leakage = True
        
    if val_test_overlap:
        warnings.warn(
            f"DATA LEAKAGE DETECTED: {len(val_test_overlap)} images appear in both "
            f"validation and test sets. "
            f"Example overlapping images: {list(val_test_overlap)[:3]}",
            UserWarning
        )
        has_leakage = True
    
    # Check if same image directory is used for different splits
    train_dir = train_dataset.image_dir
    val_dir = val_dataset.image_dir
    test_dir = test_dataset.image_dir
    
    if train_dir == val_dir:
        warnings.warn(
            f"DATA LEAKAGE DETECTED: Train and validation sets use the same image "
            f"directory: {train_dir}. This will cause data leakage.",
            UserWarning
        )
        has_leakage = True
        
    if train_dir == test_dir:
        warnings.warn(
            f"DATA LEAKAGE DETECTED: Train and test sets use the same image "
            f"directory: {train_dir}. This will cause data leakage.",
            UserWarning
        )
        has_leakage = True

    if val_dir == test_dir:
        warnings.warn(
            f"DATA LEAKAGE DETECTED: Validation and test sets use the same image "
            f"directory: {val_dir}. This will cause data leakage.",
            UserWarning
        )
        has_leakage = True
    
    return not has_leakage


def get_dataloaders(
    data_root: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    use_weighted_sampling: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        data_root: Root directory containing the ISIC 2017 dataset.
        batch_size: Batch size for all loaders.
        image_size: Target image size (height, width).
        num_workers: Number of workers for data loading.
        use_weighted_sampling: Use weighted random sampling for training.
        pin_memory: Pin memory for faster GPU transfer.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create datasets
    train_dataset = ISIC2017Dataset(
        data_root=data_root,
        split="train",
        image_size=image_size,
    )

    val_dataset = ISIC2017Dataset(
        data_root=data_root,
        split="val",
        image_size=image_size,
    )

    test_dataset = ISIC2017Dataset(
        data_root=data_root,
        split="test",
        image_size=image_size,
    )

    # Validate no data leakage between splits
    validate_no_data_leakage(train_dataset, val_dataset, test_dataset)

    # Create sampler for imbalanced training data
    train_sampler = None
    shuffle = True

    if use_weighted_sampling and len(train_dataset) > 0:
        class_weights = train_dataset.get_class_weights()
        sample_weights = [class_weights[label] for _, label in train_dataset.samples]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        shuffle = False  # Sampler handles shuffling

    # Handle empty dataset case - disable shuffle to avoid sampler issues
    if len(train_dataset) == 0:
        shuffle = False

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_dataset) > 0,  # Only drop_last if dataset has samples
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_single_loader(
    data_root: str,
    split: str = "test",
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a single data loader for a specific split.

    Args:
        data_root: Root directory containing the ISIC 2017 dataset.
        split: Dataset split - 'train', 'val', or 'test'.
        batch_size: Batch size.
        image_size: Target image size (height, width).
        num_workers: Number of workers for data loading.
        shuffle: Whether to shuffle data.
        pin_memory: Pin memory for faster GPU transfer.

    Returns:
        DataLoader for the specified split.
    """
    dataset = ISIC2017Dataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
