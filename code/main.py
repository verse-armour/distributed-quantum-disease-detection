"""
Main entry point for Quantum Splitting CNN Disease Detection Model.

This script provides a unified interface for training and testing the model.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Quantum Splitting CNN Disease Detection Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python main.py train --data-root /path/to/isic2017 --output-dir outputs

  # Test the model
  python main.py test --data-root /path/to/isic2017 --checkpoint outputs/checkpoints/best_model.pth

  # Inference on a single image
  python main.py predict --checkpoint outputs/checkpoints/best_model.pth --image /path/to/image.jpg
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--data-root", type=str, required=True, help="Path to ISIC 2017 dataset"
    )
    train_parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Output directory"
    )
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    train_parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    train_parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Test subcommand
    test_parser = subparsers.add_parser("test", help="Evaluate the model")
    test_parser.add_argument(
        "--data-root", type=str, required=True, help="Path to ISIC 2017 dataset"
    )
    test_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    test_parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Output directory"
    )
    test_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    test_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Predict on a single image")
    predict_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    predict_parser.add_argument(
        "--image", type=str, required=True, help="Path to input image"
    )

    args = parser.parse_args()

    if args.command == "train":
        from train import train

        train(
            data_root=args.data_root,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            resume=args.resume,
            seed=args.seed,
        )

    elif args.command == "test":
        from test import test

        test(
            data_root=args.data_root,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    elif args.command == "predict":
        from predict import predict_single_image

        result = predict_single_image(
            checkpoint_path=args.checkpoint,
            image_path=args.image,
        )
        print(f"\nPrediction: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nClass Probabilities:")
        for class_name, prob in result["probabilities"].items():
            print(f"  {class_name}: {prob:.2%}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
