"""
Configuration settings for Quantum Splitting CNN Disease Detection Model.

Based on experimental settings from the paper:
"Quantum Splitting Convolutional Neural Network-Based Distributed Quantum Disease Detection Model"
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR / "data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", BASE_DIR / "outputs")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", BASE_DIR / "checkpoints")

# ISIC 2017 Dataset settings
ISIC2017_CONFIG = {
    "image_size": (224, 224),  # Standard input size for MobileNetV2
    "num_classes": 3,  # melanoma, seborrheic keratosis, benign keratosis
    "class_names": ["melanoma", "seborrheic_keratosis", "benign"],
    "train_dir": "ISIC-2017_Training_Data",
    "train_gt_dir": "ISIC-2017_Training_Part1_GroundTruth",
    "val_dir": "ISIC-2017_Validation_Data",
    "val_gt_dir": "ISIC-2017_Validation_Part1_GroundTruth",
    "test_dir": "ISIC-2017_Test_v2_Data",
    "test_gt_dir": "ISIC-2017_Test_v2_Part1_GroundTruth",
    "label_file_train": "ISIC-2017_Training_Part3_GroundTruth.csv",
    "label_file_val": "ISIC-2017_Validation_Part3_GroundTruth.csv",
    "label_file_test": "ISIC-2017_Test_v2_Part3_GroundTruth.csv",
}

# Model settings
MODEL_CONFIG = {
    "n_qubits_1": 4,  # Number of qubits in first quantum circuit
    "n_qubits_2": 5,  # Number of qubits in second quantum circuit
    "n_layers": 1,  # Number of quantum circuit layers
    "classical_backbone": "mobilenetv2",  # Classical feature extractor
    "num_classes": 3,  # Output classes
    "dropout": 0.2,  # Dropout rate
}

# Training settings (consistent with paper)
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "epochs": 100,
    "early_stopping_patience": 15,
    "scheduler": "cosine",  # Learning rate scheduler
    "optimizer": "adam",
    "warmup_epochs": 5,
}

# Data augmentation settings
AUGMENTATION_CONFIG = {
    "train": {
        "random_horizontal_flip": True,
        "random_vertical_flip": True,
        "random_rotation": 30,  # degrees
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        },
        "random_resized_crop": True,
    },
    "val_test": {
        "center_crop": True,
    },
}

# Normalization values (ImageNet statistics)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Device configuration
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Random seed for reproducibility
RANDOM_SEED = 42

# Logging configuration
LOGGING_CONFIG = {
    "log_interval": 10,  # Log every N batches
    "save_interval": 5,  # Save checkpoint every N epochs
    "tensorboard": True,
}
