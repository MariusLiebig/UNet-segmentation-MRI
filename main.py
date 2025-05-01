import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch import optim
from torch.amp import GradScaler
from PIL import Image

import os
import gc

from config import base_path
from utils import (
    data_loader2D,
    data_loader3D,
    load_paths,
    save_predictions_as_img,
    mask_to_class
)
from train import Trainer
from model import UNET, UNET3D
from metric import DiceLoss, CombinedLoss


def main():
    learning_rate = 5e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    num_epochs = 30
    train_set_size = 0.8
    image_height = 512
    image_width = 512
    keep_background_fraction = 0.1 

    augmentation = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[ 0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            A.Lambda(mask=mask_to_class),  
            ToTensorV2(),
        ])
    val_augmentation = A.Compose(
                [
            # A.Resize(height=image_height, width=image_width),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[ 0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            A.Lambda(mask=mask_to_class),  

            ToTensorV2(),
        ])

    img_paths, mask_paths = load_paths()
    # img_paths, mask_paths = img_paths[0:2], mask_paths[0:2]

    model = UNET(input_channels=1, output_channels=3).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    weights = torch.tensor([0.1, 1.0, 1.0], device='cuda') #For class imbalance
    loss_fn =  CombinedLoss(num_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    torch.backends.cudnn.benchmark = True

    
    print("-"*20,"Loading Data", "-" * 20)

 
    keep_background_fraction_list = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

    for keep_background_fraction in keep_background_fraction_list:


        train_loader, val_loader = data_loader2D(
            img_paths,
            mask_paths,
            train_augmentation = augmentation,
            val_augmentation = val_augmentation,
            batch_size = batch_size,
            train_set_size = train_set_size,
            keep_background_fraction = keep_background_fraction,   
            )

        print(f"Training with keep_background_fraction = {keep_background_fraction}")
        checkpoint_path = "checkpoints/checkpoint.pth"

        if keep_background_fraction == 0.1:
            checkpoint_path = None
            learning_rate = 1e-4
        
        print("loading checkpoint from", checkpoint_path)

        print("-"*20,"Training Data", "-" * 20)
        trainer = Trainer(batch_size, learning_rate, num_epochs, model, (train_loader, val_loader), loss_fn, optimizer, scaler, early_stop_count = 1)
        if checkpoint_path is not None:
            trainer.load_checkpoint(checkpoint_path, learning_rate=learning_rate)
        trainer.train()
        
        
        del train_loader, val_loader, trainer
        torch.cuda.empty_cache()
        gc.collect()



if __name__ == '__main__':
    main()
