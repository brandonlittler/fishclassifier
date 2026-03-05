from __future__ import annotations
from typing import Tuple

from torchvision import transforms

# computed from  Fish4Knowledge crops
FISH_MEAN = (0.47290870547294617, 0.5773816704750061, 0.5515697002410889)
FISH_STD  = (0.27802103757858276, 0.3371589183807373, 0.34011566638946533)

def build_transforms(
    img_size: int = 128,
    mean: Tuple[float, float, float] = FISH_MEAN,
    std: Tuple[float, float, float] = FISH_STD,
):


    train_transforms00000 = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),

        #  convert + normalize
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transforms00000 = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transforms00000, val_transforms00000
