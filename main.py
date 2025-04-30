
import torch
import torch.nn as nn
from torch import optim
from torch.amp import GradScaler
from PIL import Image
from monai.transforms import (
    Compose,
    ResizeD,
    NormalizeIntensity,
    ScaleIntensityD,
    ToTensorD,
    Lambda
)
import argparse
import os

from config import base_path, CONFIG
from utils import (
    data_loader2D,
    data_loader3D,
    load_paths,
    save_predictions_as_img,
    mask_to_class,
    get_2d_augmentation,
    get_3d_augmentation
)
from train import Trainer
from model import UNET, UNET3D
from metric import DiceLoss, CombinedLoss



def run_training(model_class, data_loader_fn, augmentation_fn, checkpoint_path=None):
    model = model_class(input_channels=CONFIG["input_channels"], output_channels=CONFIG["output_channels"],
                     feature_size=CONFIG["feature_sizes"])


   

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    print(f"mumber of GPUs: {torch.cuda.device_count()}")
    model = model.to(CONFIG["device"])

    augmentation = augmentation_fn()


    print("-" * 20, "Loading Data", "-" * 20)
    train_loader, val_loader = data_loader_fn(
        img_paths,
        mask_paths,
        augmentation=augmentation,
        batch_size=CONFIG["batch_size"],
        train_set_size=CONFIG["train_set_size"]
    )

    print("-" * 20, "Training Data", "-" * 20)

    loss_fn = CombinedLoss(num_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=3)
    steps = len(train_loader) * CONFIG["num_epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,        # a bit higher than your current 1e-4
        total_steps=steps,
        pct_start=0.3,
        anneal_strategy="cos",
    )
    
    scaler = GradScaler()

     # Load saved weights
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=CONFIG["device"])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")

        # IMPORTANT: Reset learning rate explicitly after loading
        for param_group in optimizer.param_groups:
            param_group['lr'] = CONFIG["learning_rate"]
        print(f"Learning rate manually reset to {CONFIG['learning_rate']}")

    trainer = Trainer(CONFIG["batch_size"], CONFIG["learning_rate"], CONFIG["num_epochs"],
                      model, (train_loader, val_loader), loss_fn, optimizer, scaler, scheduler)
    trainer.train()



if __name__ == '__main__':
    #Can start 2D or 3D training from terminal
    #python main.py --d2 for 2D training
    #python main.py for 3D training
    parser = argparse.ArgumentParser()
    parser.add_argument('--d2', action='store_true', help='Use 2D segmentation instead of 3D')
    args = parser.parse_args()

    img_paths, mask_paths = load_paths()
    # img_paths, mask_paths = img_paths[0:5], mask_paths[0:5]

    if args.d2:
        run_training(UNET, data_loader2D, get_2d_augmentation)
    else:
        run_training(UNET3D, data_loader3D, get_3d_augmentation,"checkpoints/checkpoint_epoch_10.pth" )
