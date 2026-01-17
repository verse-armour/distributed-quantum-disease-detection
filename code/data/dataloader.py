"""
DataLoader utilities for ISIC 2017 dataset.

Provides functions to create train, validation, and test data loaders.
"""

from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import ISIC2017Dataset


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
