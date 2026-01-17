"""
Unit tests for utility functions.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))


class TestSetSeed:
    """Tests for random seed setting."""

    def test_set_seed_reproducibility(self):
        """Test that set_seed produces reproducible results."""
        from utils.helpers import set_seed

        set_seed(42)
        random1 = torch.rand(5)
        np_random1 = np.random.rand(5)

        set_seed(42)
        random2 = torch.rand(5)
        np_random2 = np.random.rand(5)

        assert torch.allclose(random1, random2)
        assert np.allclose(np_random1, np_random2)


class TestGetDevice:
    """Tests for device selection."""

    def test_get_device_returns_device(self):
        """Test that get_device returns a valid device."""
        from utils.helpers import get_device

        device = get_device()

        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "cpu"]


class TestCountParameters:
    """Tests for parameter counting."""

    def test_count_parameters(self):
        """Test parameter counting function."""
        from utils.helpers import count_parameters

        model = nn.Linear(10, 5)  # 10*5 + 5 = 55 parameters
        count = count_parameters(model)

        assert count == 55

    def test_count_parameters_frozen(self):
        """Test that frozen parameters are not counted."""
        from utils.helpers import count_parameters

        model = nn.Linear(10, 5)
        for param in model.parameters():
            param.requires_grad = False

        count = count_parameters(model)
        assert count == 0


class TestCheckpoints:
    """Tests for checkpoint save/load."""

    def test_save_checkpoint(self):
        """Test saving a checkpoint."""
        from utils.helpers import save_checkpoint

        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoints", "test.pth")
            save_checkpoint(model, optimizer, epoch=5, loss=0.5, accuracy=0.9, path=path)

            assert os.path.exists(path)

    def test_load_checkpoint(self):
        """Test loading a checkpoint."""
        from utils.helpers import save_checkpoint, load_checkpoint

        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pth")
            save_checkpoint(model, optimizer, epoch=5, loss=0.5, accuracy=0.9, path=path)

            # Load into new model
            new_model = nn.Linear(10, 5)
            epoch, loss, accuracy = load_checkpoint(path, new_model)

            assert epoch == 5
            assert loss == 0.5
            assert accuracy == 0.9


class TestEarlyStopping:
    """Tests for early stopping."""

    def test_early_stopping_patience(self):
        """Test early stopping with patience."""
        from utils.helpers import EarlyStopping

        early_stop = EarlyStopping(patience=3, mode="max")

        # Improving
        assert not early_stop(0.5)
        assert not early_stop(0.6)
        assert not early_stop(0.7)

        # Not improving
        assert not early_stop(0.6)  # counter = 1
        assert not early_stop(0.6)  # counter = 2
        assert early_stop(0.6)  # counter = 3, should stop

    def test_early_stopping_reset_on_improvement(self):
        """Test that counter resets on improvement."""
        from utils.helpers import EarlyStopping

        early_stop = EarlyStopping(patience=3, mode="max")

        assert not early_stop(0.5)
        assert not early_stop(0.4)  # counter = 1
        assert not early_stop(0.4)  # counter = 2
        assert not early_stop(0.6)  # improvement, counter = 0
        assert not early_stop(0.5)  # counter = 1
        assert not early_stop(0.5)  # counter = 2
        assert early_stop(0.5)  # counter = 3, should stop


class TestAverageMeter:
    """Tests for average meter."""

    def test_average_meter_update(self):
        """Test average meter updates correctly."""
        from utils.helpers import AverageMeter

        meter = AverageMeter()

        meter.update(1.0, n=2)
        meter.update(2.0, n=2)

        assert meter.avg == 1.5
        assert meter.count == 4
        assert meter.sum == 6.0

    def test_average_meter_reset(self):
        """Test average meter reset."""
        from utils.helpers import AverageMeter

        meter = AverageMeter()
        meter.update(5.0)
        meter.reset()

        assert meter.avg == 0
        assert meter.count == 0


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_compute_metrics(self):
        """Test computing classification metrics."""
        from utils.metrics import compute_metrics

        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 2, 2, 2])

        metrics = compute_metrics(y_true, y_pred, num_classes=3)

        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert metrics["accuracy"] == 5 / 6

    def test_compute_confusion_matrix(self):
        """Test confusion matrix computation."""
        from utils.metrics import compute_confusion_matrix

        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3, normalize=True)

        assert cm.shape == (3, 3)
        # Perfect predictions should have 1.0 on diagonal
        assert np.allclose(np.diag(cm), 1.0)

    def test_classification_report(self):
        """Test classification report generation."""
        from utils.metrics import get_classification_report

        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        report = get_classification_report(y_true, y_pred)

        assert isinstance(report, str)
        assert "precision" in report
        assert "recall" in report

    def test_sensitivity_specificity(self):
        """Test sensitivity and specificity computation."""
        from utils.metrics import compute_sensitivity_specificity

        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        sensitivity, specificity = compute_sensitivity_specificity(y_true, y_pred, 3)

        assert len(sensitivity) == 3
        assert len(specificity) == 3
        # Perfect predictions
        for i in range(3):
            assert sensitivity[i] == 1.0
            assert specificity[i] == 1.0
