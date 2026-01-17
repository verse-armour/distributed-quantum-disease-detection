"""
Single image prediction for Quantum Splitting CNN Disease Detection Model.
"""

import sys
from pathlib import Path
from typing import Dict

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backbone_3 import QCNet
from data import load_and_preprocess_image, get_val_transforms
from utils import get_device, load_checkpoint


CLASS_NAMES = ["melanoma", "seborrheic_keratosis", "benign"]


def predict_single_image(
    checkpoint_path: str,
    image_path: str,
    image_size: tuple = (224, 224),
) -> Dict:
    """
    Predict on a single image.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        image_path: Path to input image.
        image_size: Target image size.

    Returns:
        Dictionary with prediction results.
    """
    device = get_device()

    # Load model
    model = QCNet()
    model = model.to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    # Load and preprocess image
    transform = get_val_transforms(image_size)
    image = load_and_preprocess_image(image_path, transform)
    image = image.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]
        predicted_class = probs.argmax().item()

    # Format results
    probabilities = {
        CLASS_NAMES[i]: probs[i].item() for i in range(len(CLASS_NAMES))
    }

    return {
        "class_id": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": probs[predicted_class].item(),
        "probabilities": probabilities,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict on a single image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Image path")

    args = parser.parse_args()

    result = predict_single_image(args.checkpoint, args.image)

    print(f"\nPrediction: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nClass Probabilities:")
    for class_name, prob in result["probabilities"].items():
        print(f"  {class_name}: {prob:.2%}")
