"""
Unit tests for model components.

Tests the model architecture, tensor dimensions, and forward pass.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))


class TestMobileNetV2:
    """Tests for MobileNetV2 classical backbone."""

    def test_mobilenet_initialization(self):
        """Test MobileNetV2 model initialization."""
        from mobilnet import MobileNetV2

        model = MobileNetV2(num_classes=8)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_mobilenet_output_shape(self):
        """Test MobileNetV2 output tensor shape."""
        from mobilnet import MobileNetV2

        model = MobileNetV2(num_classes=8)
        model.eval()

        # Create dummy input: [batch_size, channels, height, width]
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        # Output should be [batch_size, num_classes=8]
        assert output.shape == (batch_size, 8), f"Expected shape (2, 8), got {output.shape}"

    def test_mobilenet_different_batch_sizes(self):
        """Test MobileNetV2 with different batch sizes."""
        from mobilnet import MobileNetV2

        model = MobileNetV2(num_classes=8)
        model.eval()

        for batch_size in [1, 4, 8]:
            dummy_input = torch.randn(batch_size, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            assert output.shape == (batch_size, 8)


class TestQNN:
    """Tests for Quantum Neural Network component."""

    def test_qnn_initialization(self):
        """Test QNN model initialization."""
        from mps3 import QNN

        model = QNN()
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_qnn_output_shape(self):
        """Test QNN output tensor shape."""
        from mps3 import QNN

        model = QNN()
        model.eval()

        # QNN expects input of shape [batch_size, 8]
        # (n_qubits_1=4 + n_qubits_2=5 with overlap of 1 qubit = 8 total)
        batch_size = 2
        dummy_input = torch.randn(batch_size, 8)

        with torch.no_grad():
            output = model(dummy_input)

        # Output should be [batch_size, 2^n_qubits_2] = [batch_size, 32]
        assert output.shape == (batch_size, 32), f"Expected shape (2, 32), got {output.shape}"

    def test_qnn_single_sample(self):
        """Test QNN with single sample."""
        from mps3 import QNN

        model = QNN()
        model.eval()

        dummy_input = torch.randn(1, 8)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (1, 32)


class TestQCNet:
    """Tests for the full QCNet (Quantum-Classical hybrid) model."""

    def test_qcnet_initialization(self):
        """Test QCNet model initialization."""
        from backbone_3 import QCNet

        model = QCNet()
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_qcnet_components(self):
        """Test QCNet contains all expected components."""
        from backbone_3 import QCNet

        model = QCNet()

        # Check components exist
        assert hasattr(model, "CModel"), "Missing classical model (CModel)"
        assert hasattr(model, "QModel"), "Missing quantum model (QModel)"
        assert hasattr(model, "fc2"), "Missing final fully connected layer (fc2)"

    def test_qcnet_output_shape(self):
        """Test QCNet end-to-end output shape."""
        from backbone_3 import QCNet

        model = QCNet()
        model.eval()

        # Input: [batch_size, channels, height, width]
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        # Output should be [batch_size, num_classes=3]
        assert output.shape == (batch_size, 3), f"Expected shape (2, 3), got {output.shape}"

    def test_qcnet_tensor_dimensions_match(self):
        """Test that tensor dimensions match between layers."""
        from backbone_3 import QCNet

        model = QCNet()
        model.eval()

        dummy_input = torch.randn(2, 3, 224, 224)

        # Test classical model output
        with torch.no_grad():
            classical_output = model.CModel(dummy_input)

        assert classical_output.shape[1] == 8, "Classical model should output 8 features"

        # Test quantum model can accept classical output
        with torch.no_grad():
            quantum_output = model.QModel(classical_output)

        assert quantum_output.shape[1] == 32, "QNN should output 32 features"

        # Test final layer can accept quantum output
        with torch.no_grad():
            final_output = model.fc2(quantum_output)

        assert final_output.shape[1] == 3, "Final layer should output 3 classes"

    def test_qcnet_forward_pass_no_error(self):
        """Test that forward pass completes without error."""
        from backbone_3 import QCNet

        model = QCNet()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)

        try:
            with torch.no_grad():
                output = model(dummy_input)
            success = True
        except Exception as e:
            success = False
            pytest.fail(f"Forward pass failed with error: {e}")

        assert success

    def test_qcnet_backward_pass(self):
        """Test that backward pass works for training."""
        from backbone_3 import QCNet

        model = QCNet()
        model.train()

        dummy_input = torch.randn(2, 3, 224, 224, requires_grad=True)
        dummy_labels = torch.tensor([0, 1])

        output = model(dummy_input)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, dummy_labels)

        # Should be able to compute gradients
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break

        assert has_grad, "No gradients computed during backward pass"


class TestNoSplitQNN:
    """Tests for QNN without circuit cutting (no_cut_qnn)."""

    def test_no_cut_qnn_initialization(self):
        """Test non-split QNN model initialization."""
        from no_cut_qnn import QNN

        model = QNN()
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_no_cut_qnn_output_shape(self):
        """Test non-split QNN output shape."""
        from no_cut_qnn import QNN

        model = QNN()
        model.eval()

        # This QNN uses n_qubits_1=8
        batch_size = 2
        dummy_input = torch.randn(batch_size, 8)

        with torch.no_grad():
            output = model(dummy_input)

        # Output should be [batch_size, 2^5] for 5 measured qubits
        assert output.shape[0] == batch_size
