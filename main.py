import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch import optim
from torch.amp import GradScaler
from PIL import Image

import os

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
    learning_rate = 1e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    num_epochs = 30
    train_set_size = 0.8
    image_height = 512
    image_width = 512
    keep_background_fraction = 0.3

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

    img_paths, mask_paths = load_paths()
    # img_paths, mask_paths = img_paths[0:2], mask_paths[0:2]

    model = UNET(input_channels=1, output_channels=3).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    weights = torch.tensor([0.1, 1.0, 1.0], device='cuda') #For class imbalance
    loss_fn =  CombinedLoss(num_classes=3)
    # loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("-"*20,"Loading Data", "-" * 20)
    train_loader, val_loader = data_loader2D(
        img_paths,
        mask_paths,
        augmentation = augmentation,
        batch_size = batch_size,
        train_set_size = train_set_size,
        keep_background_fraction = keep_background_fraction,   
        )

    # Load model?????
    # if LOAD_MODEL:
    checkpoint_path = "checkpoints/checkpoint_epoch_13.pth"
    # checkpoint_path = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cuda",weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # IMPORTANT: Reset learning rate explicitly after loading


    # check_accuracy(val_loader, model, device=DEVICE)
    scaler = GradScaler()

    print("-"*20,"Training Data", "-" * 20)
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(batch_size, learning_rate, num_epochs, model, (train_loader, val_loader), loss_fn, optimizer, scaler)
    trainer.train()

#Plots and accuracy

def test():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    train_set_size = 0.8


    augmentation = A.Compose(
        [
            # A.Resize(height=image_height, width=image_width),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            # A.Normalize(
            #     mean=[0.0],
            #     std=[1.0],
            #     max_pixel_value=255.0,
            # ),
            A.Lambda(mask=mask_to_class),  
            ToTensorV2(),
        ])
    

    img_paths, mask_paths = load_paths()
    img_paths, mask_paths = img_paths[0:2], mask_paths[0:2]

    model = UNET(input_channels=1, output_channels=3).to(DEVICE)
    
    print("-"*20,"Loading Data", "-" * 20)
    train_loader, val_loader = data_loader2D(
        img_paths,
        mask_paths,
        augmentation = augmentation,
        batch_size = batch_size,
        train_set_size = train_set_size,
        )
    print("-"*20,"Testing Data", "-" * 20)
    save_predictions_as_img(
        val_loader,
        model,
        folder="saved_images_test2/",
        device=DEVICE,
        max_examples=40,
    )



if __name__ == '__main__':
    main()
