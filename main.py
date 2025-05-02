import argparse
import gc
import torch
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler
from torch import nn, optim
import cv2

#Own classes and functions
from config import CONFIG
from utils import (
    data_loader2D,
    load_paths,
    mask_to_class
)
from model import UNET
from train import Trainer
from model import UNET
from metric import CombinedLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Setting keep_background_fraction from command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep_background_fraction", type=float, default=0.1)
    args = parser.parse_args()
    CONFIG["keep_background_fraction"] = args.keep_background_fraction


    # Loading checkpoint from earlier training with different keep_background_fraction
    old_fractions = [0.0, 0.0, 0.1, 0.2, 0.3, 0.3, 0.5, 0.5, 0.5, 0.8, 0.8]
    fraction_str = int(CONFIG['keep_background_fraction'] * 10)
    checkpoint_paths = glob.glob(f"checkpoints/best_checkpoint_{old_fractions[fraction_str]}*.pth")
    if CONFIG["keep_background_fraction"] == 0.1:
        checkpoint_path = None
        CONFIG["learning_rate"] = 1e-4
    else:
        checkpoint_path = sorted(checkpoint_paths)[-1] if checkpoint_paths else None



    # Data augmentation for training and validation
    augmentation = A.Compose([
        A.Resize(height=CONFIG["image_height"], width=CONFIG["image_width"]),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.05,  # small spatial shifts
            scale_limit=0.1,   # slight zoom in/out
            rotate_limit=15,   # limited rotation to mimic patient orientation
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),

        # Intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

        # Normalization and conversion
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        A.Lambda(mask=mask_to_class),
        ToTensorV2(),
    ])
    val_augmentation = A.Compose([
        A.Resize(height=CONFIG["image_height"], width=CONFIG["image_width"]),
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        A.Lambda(mask=mask_to_class),
        ToTensorV2(),
    ])



    #Loading data
    print("-" * 20, "Loading Data", "-" * 20)
    print(f"keep_background_fraction = {CONFIG['keep_background_fraction']}")
    img_paths, mask_paths = load_paths()
    img_paths, mask_paths = img_paths[0:2], mask_paths[0:2]
    train_loader, val_loader = data_loader2D(
        img_paths,
        mask_paths,
        train_augmentation=augmentation,
        val_augmentation=val_augmentation,
        batch_size=CONFIG["batch_size"],
        train_set_size=CONFIG["train_set_size"],
        keep_background_fraction=CONFIG["keep_background_fraction"]
    )



    # Loading model
    print("-" * 20, "Loading model", "-" * 20)
    model = UNET(input_channels=1, output_channels=3).to(DEVICE)
    loss_fn = CombinedLoss(num_classes=CONFIG["num_classes"],  weights = torch.tensor(CONFIG["weights"], device='cuda'), dice_weight=CONFIG["dice_weight"], ce_weight=CONFIG["ce_weight"]) #Own loss function
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scaler = GradScaler()
    torch.backends.cudnn.benchmark = True

    print(f"Training with keep_background_fraction = {CONFIG['keep_background_fraction']}")
    trainer = Trainer(
        CONFIG["batch_size"],
        CONFIG["learning_rate"],
        CONFIG["num_epochs"],
        model,
        (train_loader, val_loader),
        loss_fn,
        optimizer,
        scaler,
        CONFIG["keep_background_fraction"],
        early_stop_count=CONFIG["early_stop_count"],
        checkpoint_path=checkpoint_path,
    )

    print("-" * 20, "Training", "-" * 20)
    trainer.train()



if __name__ == '__main__':
    main()
