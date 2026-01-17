"""
ISIC 2017 Skin Cancer Dataset loader.

Supports loading training, validation, and test splits with corresponding labels.
Task 3: Lesion diagnosis (3-class classification: melanoma, seborrheic keratosis, benign)
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Dict

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .preprocessing import get_train_transforms, get_val_transforms


class ISIC2017Dataset(Dataset):
    """
    ISIC 2017 Skin Cancer Dataset for Task 3 (Lesion Classification).

    Directory structure expected:
        data_root/
            ISIC-2017_Training_Data/
                ISIC_0000000.jpg
                ...
            ISIC-2017_Training_Part3_GroundTruth.csv
            ISIC-2017_Validation_Data/
                ...
            ISIC-2017_Validation_Part3_GroundTruth.csv
            ISIC-2017_Test_v2_Data/
                ...
            ISIC-2017_Test_v2_Part3_GroundTruth.csv

    Label CSV format (Task 3):
        image_id, melanoma, seborrheic_keratosis
        ISIC_0000000, 0.0, 0.0  (benign)
        ISIC_0000001, 1.0, 0.0  (melanoma)
        ISIC_0000002, 0.0, 1.0  (seborrheic keratosis)
    """

    CLASS_NAMES = ["melanoma", "seborrheic_keratosis", "benign"]
    NUM_CLASSES = 3

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (224, 224),
    ):
        """
        Initialize ISIC 2017 dataset.

        Args:
            data_root: Root directory containing the dataset.
            split: Dataset split - 'train', 'val', or 'test'.
            transform: Optional custom transforms to apply.
            image_size: Target image size (height, width).
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size

        # Set up paths based on split
        if split == "train":
            self.image_dir = self.data_root / "ISIC-2017_Training_Data"
            self.label_file = self.data_root / "ISIC-2017_Training_Part3_GroundTruth.csv"
        elif split == "val":
            self.image_dir = self.data_root / "ISIC-2017_Validation_Data"
            self.label_file = self.data_root / "ISIC-2017_Validation_Part3_GroundTruth.csv"
        elif split == "test":
            self.image_dir = self.data_root / "ISIC-2017_Test_v2_Data"
            self.label_file = self.data_root / "ISIC-2017_Test_v2_Part3_GroundTruth.csv"
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

        # Set up transforms
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        # Load labels
        self.samples = self._load_labels()

    def _load_labels(self) -> List[Tuple[str, int]]:
        """
        Load image paths and labels from CSV file.

        Returns:
            List of (image_path, label) tuples.
        """
        samples = []

        if not self.label_file.exists():
            # Return empty list if label file doesn't exist (for testing without data)
            return samples

        df = pd.read_csv(self.label_file)

        for _, row in df.iterrows():
            image_id = row["image_id"]

            # Determine label based on columns
            # melanoma = 1, seborrheic_keratosis = 1, else benign
            if row.get("melanoma", 0) == 1.0:
                label = 0  # melanoma
            elif row.get("seborrheic_keratosis", 0) == 1.0:
                label = 1  # seborrheic_keratosis
            else:
                label = 2  # benign

            # Find image file (handle different extensions)
            image_path = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                potential_path = self.image_dir / f"{image_id}{ext}"
                if potential_path.exists():
                    image_path = potential_path
                    break

            if image_path is not None:
                samples.append((str(image_path), label))

        return samples

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label).
        """
        image_path, label = self.samples[idx]

        # Load and transform image
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of classes in the dataset.

        Returns:
            Dictionary mapping class names to counts.
        """
        counts = {name: 0 for name in self.CLASS_NAMES}

        for _, label in self.samples:
            counts[self.CLASS_NAMES[label]] += 1

        return counts

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced data.

        Returns:
            Tensor of class weights (inverse frequency).
        """
        distribution = self.get_class_distribution()
        total = sum(distribution.values())

        if total == 0:
            return torch.ones(self.NUM_CLASSES)

        weights = []
        for name in self.CLASS_NAMES:
            count = distribution[name]
            if count > 0:
                weights.append(total / (self.NUM_CLASSES * count))
            else:
                weights.append(1.0)

        return torch.tensor(weights, dtype=torch.float32)


class ISIC2017SegmentationDataset(Dataset):
    """
    ISIC 2017 Dataset for Task 1 (Lesion Segmentation).

    Loads images with corresponding segmentation masks.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (224, 224),
    ):
        """
        Initialize ISIC 2017 segmentation dataset.

        Args:
            data_root: Root directory containing the dataset.
            split: Dataset split - 'train', 'val', or 'test'.
            image_transform: Optional custom transforms for images.
            mask_transform: Optional custom transforms for masks.
            image_size: Target image size (height, width).
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size

        # Set up paths based on split
        if split == "train":
            self.image_dir = self.data_root / "ISIC-2017_Training_Data"
            self.mask_dir = self.data_root / "ISIC-2017_Training_Part1_GroundTruth"
        elif split == "val":
            self.image_dir = self.data_root / "ISIC-2017_Validation_Data"
            self.mask_dir = self.data_root / "ISIC-2017_Validation_Part1_GroundTruth"
        elif split == "test":
            self.image_dir = self.data_root / "ISIC-2017_Test_v2_Data"
            self.mask_dir = self.data_root / "ISIC-2017_Test_v2_Part1_GroundTruth"
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

        # Set up transforms
        if image_transform is not None:
            self.image_transform = image_transform
        elif split == "train":
            self.image_transform = get_train_transforms(image_size)
        else:
            self.image_transform = get_val_transforms(image_size)

        self.mask_transform = mask_transform

        # Load samples
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, str]]:
        """
        Load image and mask path pairs.

        Returns:
            List of (image_path, mask_path) tuples.
        """
        samples = []

        if not self.image_dir.exists():
            return samples

        for image_file in self.image_dir.iterdir():
            if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                image_id = image_file.stem
                mask_path = self.mask_dir / f"{image_id}_segmentation.png"

                if mask_path.exists():
                    samples.append((str(image_file), str(mask_path)))

        return samples

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, mask_tensor).
        """
        image_path, mask_path = self.samples[idx]

        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            from torchvision import transforms
            mask = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])(mask)

        return image, mask
