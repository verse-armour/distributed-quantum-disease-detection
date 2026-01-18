"""
Unit tests for data loading and preprocessing.
"""

import sys
import tempfile
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from data.dataset import ISIC2017Dataset
from data.dataloader import validate_no_data_leakage


def create_dummy_image(size: Tuple[int, int] = (300, 300)) -> Image.Image:
    """Create a dummy RGB image for testing."""
    array = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_train_transforms_output_shape(self):
        """Test that train transforms produce correct tensor shape."""
        from data.preprocessing import get_train_transforms

        transform = get_train_transforms(image_size=(224, 224))
        image = create_dummy_image()

        tensor = transform(image)

        assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"

    def test_val_transforms_output_shape(self):
        """Test that validation transforms produce correct tensor shape."""
        from data.preprocessing import get_val_transforms

        transform = get_val_transforms(image_size=(224, 224))
        image = create_dummy_image()

        tensor = transform(image)

        assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"

    def test_transforms_normalized(self):
        """Test that transforms produce normalized values."""
        from data.preprocessing import get_val_transforms

        transform = get_val_transforms(image_size=(224, 224))
        image = create_dummy_image()

        tensor = transform(image)

        # After normalization, values should not be in [0, 255]
        # They should be roughly in [-2, 3] range for ImageNet normalization
        assert tensor.min() < 0 or tensor.max() < 1, "Tensor appears unnormalized"

    def test_denormalize(self):
        """Test denormalization function."""
        from data.preprocessing import get_val_transforms, denormalize

        transform = get_val_transforms(image_size=(224, 224))
        image = create_dummy_image()

        tensor = transform(image)
        denorm = denormalize(tensor)

        # After denormalization, values should be roughly in [0, 1]
        assert denorm.min() >= -0.1, f"Min value {denorm.min()} too low"
        assert denorm.max() <= 1.1, f"Max value {denorm.max()} too high"

    def test_load_and_preprocess_image(self):
        """Test loading and preprocessing a single image."""
        from data.preprocessing import load_and_preprocess_image

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image = create_dummy_image()
            image.save(f.name)

            tensor = load_and_preprocess_image(f.name, image_size=(224, 224))

            assert tensor.shape == (3, 224, 224)
            assert isinstance(tensor, torch.Tensor)

    def test_different_image_sizes(self):
        """Test transforms with different target sizes."""
        from data.preprocessing import get_val_transforms

        for size in [(128, 128), (224, 224), (256, 256)]:
            transform = get_val_transforms(image_size=size)
            image = create_dummy_image()

            tensor = transform(image)

            assert tensor.shape == (3, size[0], size[1])


class TestISIC2017Dataset:
    """Tests for ISIC 2017 dataset loader."""

    def test_dataset_initialization_empty(self):
        """Test dataset initialization with non-existent path."""
        from data.dataset import ISIC2017Dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = ISIC2017Dataset(data_root=tmpdir, split="train")

            # Should initialize without error, but be empty
            assert len(dataset) == 0

    def test_dataset_class_names(self):
        """Test that class names are correctly defined."""
        from data.dataset import ISIC2017Dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = ISIC2017Dataset(data_root=tmpdir, split="train")

            assert dataset.CLASS_NAMES == ["melanoma", "seborrheic_keratosis", "benign"]
            assert dataset.NUM_CLASSES == 3

    def test_dataset_invalid_split(self):
        """Test that invalid split raises error."""
        from data.dataset import ISIC2017Dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Invalid split"):
                ISIC2017Dataset(data_root=tmpdir, split="invalid")

    def test_dataset_class_weights_empty(self):
        """Test class weights for empty dataset."""
        from data.dataset import ISIC2017Dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = ISIC2017Dataset(data_root=tmpdir, split="train")
            weights = dataset.get_class_weights()

            assert weights.shape == (3,)
            assert torch.allclose(weights, torch.ones(3))


class TestDataloader:
    """Tests for dataloader utilities."""

    def test_get_single_loader(self):
        """Test creating a single data loader."""
        from data.dataloader import get_single_loader

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = get_single_loader(
                data_root=tmpdir,
                split="test",
                batch_size=4,
            )

            assert loader is not None
            assert loader.batch_size == 4

    def test_get_dataloaders(self):
        """Test creating all data loaders."""
        from data.dataloader import get_dataloaders

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use weighted_sampling=False for empty datasets to avoid sampler error
            train_loader, val_loader, test_loader = get_dataloaders(
                data_root=tmpdir,
                batch_size=4,
                use_weighted_sampling=False,  # Avoid sampler error with empty dataset
            )

            assert train_loader is not None
            assert val_loader is not None
            assert test_loader is not None


class TestMaskTransform:
    """Tests for segmentation mask transforms."""

    def test_mask_transform(self):
        """Test mask transform for segmentation."""
        from data.preprocessing import get_mask_transform

        transform = get_mask_transform(image_size=(224, 224))

        # Create dummy mask (grayscale)
        mask_array = np.random.randint(0, 2, (300, 300), dtype=np.uint8) * 255
        mask = Image.fromarray(mask_array, mode="L")

        tensor = transform(mask)

        assert tensor.shape == (1, 224, 224)
        assert tensor.dtype == torch.float32


class TestDataLeakageValidation:
    """Tests for data leakage validation function."""

    def test_no_data_leakage(self):
        """Test that no warning is raised when there is no data leakage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure with different images for each split
            train_dir = Path(tmpdir) / "ISIC-2017_Training_Data"
            val_dir = Path(tmpdir) / "ISIC-2017_Validation_Data"
            test_dir = Path(tmpdir) / "ISIC-2017_Test_v2_Data"
            train_dir.mkdir()
            val_dir.mkdir()
            test_dir.mkdir()

            # Create different images for each split
            for i in range(3):
                img = create_dummy_image()
                img.save(train_dir / f"ISIC_{i:07d}.jpg")
                img.save(val_dir / f"ISIC_{i+100:07d}.jpg")
                img.save(test_dir / f"ISIC_{i+200:07d}.jpg")

            # Create label files with different image IDs
            pd.DataFrame({
                "image_id": [f"ISIC_{i:07d}" for i in range(3)],
                "melanoma": [1.0, 0.0, 0.0],
                "seborrheic_keratosis": [0.0, 1.0, 0.0],
            }).to_csv(Path(tmpdir) / "ISIC-2017_Training_Part3_GroundTruth.csv", index=False)

            pd.DataFrame({
                "image_id": [f"ISIC_{i+100:07d}" for i in range(3)],
                "melanoma": [0.0, 1.0, 0.0],
                "seborrheic_keratosis": [0.0, 0.0, 1.0],
            }).to_csv(Path(tmpdir) / "ISIC-2017_Validation_Part3_GroundTruth.csv", index=False)

            pd.DataFrame({
                "image_id": [f"ISIC_{i+200:07d}" for i in range(3)],
                "melanoma": [0.0, 0.0, 1.0],
                "seborrheic_keratosis": [0.0, 0.0, 0.0],
            }).to_csv(Path(tmpdir) / "ISIC-2017_Test_v2_Part3_GroundTruth.csv", index=False)

            # Load datasets
            train_dataset = ISIC2017Dataset(tmpdir, split="train")
            val_dataset = ISIC2017Dataset(tmpdir, split="val")
            test_dataset = ISIC2017Dataset(tmpdir, split="test")

            # Validate no data leakage
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = validate_no_data_leakage(train_dataset, val_dataset, test_dataset)

            assert result is True
            assert len(w) == 0

    def test_data_leakage_detected(self):
        """Test that warning is raised when data leakage is detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure with overlapping images
            train_dir = Path(tmpdir) / "ISIC-2017_Training_Data"
            val_dir = Path(tmpdir) / "ISIC-2017_Validation_Data"
            test_dir = Path(tmpdir) / "ISIC-2017_Test_v2_Data"
            train_dir.mkdir()
            val_dir.mkdir()
            test_dir.mkdir()

            # Create same-named images for each split (simulating data leakage)
            for i in range(3):
                img = create_dummy_image()
                img.save(train_dir / f"ISIC_{i:07d}.jpg")
                img.save(val_dir / f"ISIC_{i:07d}.jpg")  # Same file name!
                img.save(test_dir / f"ISIC_{i+200:07d}.jpg")

            # Create label files with overlapping image IDs
            pd.DataFrame({
                "image_id": [f"ISIC_{i:07d}" for i in range(3)],
                "melanoma": [1.0, 0.0, 0.0],
                "seborrheic_keratosis": [0.0, 1.0, 0.0],
            }).to_csv(Path(tmpdir) / "ISIC-2017_Training_Part3_GroundTruth.csv", index=False)

            pd.DataFrame({
                "image_id": [f"ISIC_{i:07d}" for i in range(3)],  # Same image IDs!
                "melanoma": [1.0, 0.0, 0.0],
                "seborrheic_keratosis": [0.0, 1.0, 0.0],
            }).to_csv(Path(tmpdir) / "ISIC-2017_Validation_Part3_GroundTruth.csv", index=False)

            pd.DataFrame({
                "image_id": [f"ISIC_{i+200:07d}" for i in range(3)],
                "melanoma": [0.0, 0.0, 1.0],
                "seborrheic_keratosis": [0.0, 0.0, 0.0],
            }).to_csv(Path(tmpdir) / "ISIC-2017_Test_v2_Part3_GroundTruth.csv", index=False)

            # Load datasets
            train_dataset = ISIC2017Dataset(tmpdir, split="train")
            val_dataset = ISIC2017Dataset(tmpdir, split="val")
            test_dataset = ISIC2017Dataset(tmpdir, split="test")

            # Validate data leakage is detected
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = validate_no_data_leakage(train_dataset, val_dataset, test_dataset)

            assert result is False
            assert len(w) >= 1  # At least one warning should be raised
            # Check that the warning message mentions data leakage
            assert any("DATA LEAKAGE" in str(warning.message) for warning in w)
