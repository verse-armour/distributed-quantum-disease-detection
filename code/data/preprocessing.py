"""
Image preprocessing utilities for ISIC 2017 skin cancer dataset.

Provides standard preprocessing transforms for training, validation, and testing.
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List

# ImageNet normalization values
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    random_horizontal_flip: bool = True,
    random_vertical_flip: bool = True,
    random_rotation: int = 30,
    color_jitter: Optional[dict] = None,
) -> transforms.Compose:
    """
    Get training transforms with data augmentation.

    Args:
        image_size: Target image size (height, width).
        random_horizontal_flip: Apply random horizontal flip.
        random_vertical_flip: Apply random vertical flip.
        random_rotation: Maximum rotation angle in degrees.
        color_jitter: Color jitter parameters dict.

    Returns:
        Composed transforms for training.
    """
    transform_list = [
        transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
        transforms.RandomCrop(image_size),
    ]

    if random_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if random_vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))

    if random_rotation > 0:
        transform_list.append(transforms.RandomRotation(random_rotation))

    if color_jitter:
        transform_list.append(
            transforms.ColorJitter(
                brightness=color_jitter.get("brightness", 0),
                contrast=color_jitter.get("contrast", 0),
                saturation=color_jitter.get("saturation", 0),
                hue=color_jitter.get("hue", 0),
            )
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )

    return transforms.Compose(transform_list)


def get_val_transforms(
    image_size: Tuple[int, int] = (224, 224),
) -> transforms.Compose:
    """
    Get validation/test transforms without augmentation.

    Args:
        image_size: Target image size (height, width).

    Returns:
        Composed transforms for validation/testing.
    """
    return transforms.Compose(
        [
            transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def denormalize(
    tensor: torch.Tensor,
    mean: List[float] = NORMALIZE_MEAN,
    std: List[float] = NORMALIZE_STD,
) -> torch.Tensor:
    """
    Denormalize a tensor image for visualization.

    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W].
        mean: Normalization mean values.
        std: Normalization std values.

    Returns:
        Denormalized tensor.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean


def load_and_preprocess_image(
    image_path: str,
    transform: Optional[transforms.Compose] = None,
    image_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """
    Load and preprocess a single image.

    Args:
        image_path: Path to the image file.
        transform: Optional custom transform to apply.
        image_size: Target image size if no transform provided.

    Returns:
        Preprocessed image tensor.
    """
    image = Image.open(image_path).convert("RGB")

    if transform is None:
        transform = get_val_transforms(image_size)

    return transform(image)


def get_mask_transform(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Get transforms for segmentation masks.

    Args:
        image_size: Target mask size (height, width).

    Returns:
        Composed transforms for masks.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )
