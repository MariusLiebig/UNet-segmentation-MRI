import argparse
import gc
import torch
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler
from torch import nn, optim

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
    old_fractions = [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fraction_str = f"{int(CONFIG['keep_background_fraction'] * 10)}"
    checkpoint_paths = glob.glob(f"checkpoints/best_checkpoint_{fraction_str}*.pth")
    if CONFIG["keep_background_fraction"] == 0.1:
        checkpoint_path = None
        CONFIG["learning_rate"] = 1e-4
    else:
        checkpoint_path = sorted(checkpoint_paths)[-1] if checkpoint_paths else None



    # Data augmentation for training and validation
    augmentation = A.Compose([
        A.Resize(height=CONFIG["image_height"], width=CONFIG["image_width"]),
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
    img_paths, mask_paths = load_paths()
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
        early_stop_count=CONFIG["early_stop_count"]
    )
    # Loading checkpoint if available
    if checkpoint_path is not None:
        print("Loading checkpoint from", checkpoint_path)
        trainer.load_checkpoint(checkpoint_path, learning_rate=CONFIG["learning_rate"])

    trainer.train()



if __name__ == '__main__':
    main()
