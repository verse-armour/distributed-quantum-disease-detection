# Distributed Quantum Disease Detection

Quantum Splitting Convolutional Neural Network-Based Distributed Quantum Disease Detection Model

## Overview

This repository implements a hybrid quantum-classical neural network for skin cancer classification using the ISIC 2017 dataset. The model combines a MobileNetV2 classical backbone with a quantum neural network (QNN) using circuit cutting techniques for distributed quantum computing.

## Architecture

The model consists of three main components:

1. **Classical Backbone (MobileNetV2)**: Extracts 8-dimensional features from 224×224 RGB images
2. **Quantum Neural Network (QNN)**: Processes features using split quantum circuits with 4+5 qubits
3. **Classification Head**: Maps 32-dimensional quantum outputs to 3 disease classes

### Quantum Circuit Design

The quantum component uses circuit cutting to split a larger circuit into smaller subcircuits that can run on distributed quantum hardware:
- Front circuit: 4 qubits
- Back circuit: 5 qubits  
- Cut point: 1 shared qubit

## Project Structure

```
distributed-quantum-disease-detection/
├── code/
│   ├── backbone_3.py      # Main QCNet model (MobileNetV2 + QNN)
│   ├── mobilnet.py        # MobileNetV2 classical backbone
│   ├── mps3.py            # Quantum Neural Network with circuit cutting
│   ├── qnn.py             # Alternative QNN implementation
│   ├── no_cut_qnn.py      # QNN without circuit cutting
│   ├── mlp.py             # Simple MLP model
│   ├── matrix.py          # Confusion matrix visualization
│   ├── config.py          # Configuration settings
│   ├── train.py           # Training script
│   ├── test.py            # Evaluation script
│   ├── predict.py         # Single image prediction
│   ├── main.py            # Main entry point
│   ├── data/              # Data loading modules
│   │   ├── preprocessing.py  # Image preprocessing
│   │   ├── dataset.py        # ISIC 2017 dataset loader
│   │   └── dataloader.py     # DataLoader utilities
│   └── utils/             # Utility functions
│       ├── helpers.py        # Training helpers
│       └── metrics.py        # Evaluation metrics
├── tests/                 # Unit tests
│   ├── test_model.py      # Model tests
│   ├── test_data.py       # Data loading tests
│   └── test_utils.py      # Utility tests
├── requirements.txt       # Python dependencies
├── pytest.ini            # Test configuration
└── .github/workflows/    # CI configuration
    └── ci.yml
```

## Installation

```bash
# Clone the repository
git clone https://github.com/verse-armour/distributed-quantum-disease-detection.git
cd distributed-quantum-disease-detection

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup (ISIC 2017)

Download the ISIC 2017 dataset from [ISIC Archive](https://challenge.isic-archive.com/data/#2017):

1. **Training Data** (5.8GB): Lesion images
2. **Training Ground Truth** (43KB): Lesion diagnoses (Task 3)
3. **Validation Data** (878MB): Validation images
4. **Validation Ground Truth** (3KB): Validation diagnoses
5. **Test Data** (5.4GB): Test images
6. **Test Ground Truth** (13KB): Test diagnoses

Organize the dataset as follows:
```
data/
├── ISIC-2017_Training_Data/
│   └── ISIC_0000000.jpg, ...
├── ISIC-2017_Training_Part3_GroundTruth.csv
├── ISIC-2017_Validation_Data/
│   └── ISIC_0000000.jpg, ...
├── ISIC-2017_Validation_Part3_GroundTruth.csv
├── ISIC-2017_Test_v2_Data/
│   └── ISIC_0000000.jpg, ...
└── ISIC-2017_Test_v2_Part3_GroundTruth.csv
```

### Classes
- **0**: Melanoma
- **1**: Seborrheic Keratosis  
- **2**: Benign

## Usage

### Training

```bash
cd code
python main.py train --data-root /path/to/isic2017 --output-dir outputs --epochs 100
```

Options:
- `--batch-size`: Training batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--epochs`: Number of epochs (default: 100)
- `--patience`: Early stopping patience (default: 15)
- `--resume`: Resume from checkpoint

### Evaluation

```bash
python main.py test --data-root /path/to/isic2017 --checkpoint outputs/checkpoints/best_model.pth
```

### Single Image Prediction

```bash
python main.py predict --checkpoint outputs/checkpoints/best_model.pth --image /path/to/image.jpg
```

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

Run specific test modules:
```bash
pytest tests/test_model.py -v  # Model tests
pytest tests/test_data.py -v   # Data loading tests
pytest tests/test_utils.py -v  # Utility tests
```

## Model Details

### Input/Output Dimensions
- Input: `[batch_size, 3, 224, 224]` (RGB images)
- MobileNetV2 output: `[batch_size, 8]`
- QNN output: `[batch_size, 32]`
- Final output: `[batch_size, 3]` (class probabilities)

### Experimental Settings
- Image size: 224×224
- Batch size: 32
- Learning rate: 1e-4
- Optimizer: Adam with weight decay 1e-5
- Scheduler: Cosine annealing
- Early stopping patience: 15 epochs

## Citation

If you use this code, please cite the original paper:
```
Li et al., "Quantum Splitting Convolutional Neural Network-Based Distributed Quantum Disease Detection Model", 2025
```

## License

This project is provided for research purposes.
