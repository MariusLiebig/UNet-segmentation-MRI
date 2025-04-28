import os
# import monai
# from monai.transforms import LoadImage
# import matplotlib.pyplot as plt
from config import base_path
import torch
import torchvision
import glob
from PIL import Image
import numpy as np
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import monai.transforms as mt
from monai.transforms import (
    Compose,
    LoadImaged,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandBiasFieldd,
    RandGaussianNoised,
    RandSpatialCropd,
    NormalizeIntensityd,
    RandZoomd,
    ToTensord,
    CropForegroundd,
    RandCropByPosNegLabeld,
)
from data.loader import MedImgDataset2D, MedImgDataset3D
from config import CONFIG

def to_cuda(tensor):
    if torch.cuda.is_available():
        if type(tensor) == tuple or type(tensor) == list:
            return [x.cuda() for x in tensor]
        return tensor.cuda()
    return tensor


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])




def data_loader2D(image_paths, mask_paths, augmentation, batch_size, train_set_size = 0.8, keep_background_fraction = 0.1):
    #Split into train and validation set via pathdir
    full_dataset = MedImgDataset2D(image_paths, mask_paths, augmentation=augmentation, get_all_slices=True, keep_background_fraction=CONFIG["keep_background_fraction"])
    print(f"Full dataset length: {len(full_dataset)}")

    train_size = int(train_set_size * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Randomly split dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Dataloaders, train_loader -> shuffle = true, val_loader -> shuffle = false
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    return train_loader, val_loader

def data_loader3D(image_paths, mask_paths, augmentation, batch_size, train_set_size = 0.8):
    full_dataset = MedImgDataset3D(image_paths, mask_paths, augmentation=augmentation)
    print(f"Full dataset length: {len(full_dataset)}")

    train_size = int(train_set_size * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Randomly split dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Dataloaders, train_loader -> shuffle = true, val_loader -> shuffle = false
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    return train_loader, val_loader

def load_paths():

    img_pattern = os.path.join(base_path,'**/*/preRT/*_T2.nii.gz')
    mask_pattern = os.path.join(base_path,'**/*/preRT/*_mask.nii.gz')

    image_paths = glob.glob(img_pattern, recursive=True)
    mask_paths = glob.glob(mask_pattern, recursive=True)

    return image_paths, mask_paths


def mask_to_class(x, **kwargs):
    x_new = (x == 0.5).astype('uint8') + (x == 1).astype('uint8') * 2

    return x_new



def get_2d_augmentation():
    return A.Compose([
        A.Resize(height=CONFIG["image_height"], width=CONFIG["image_width"]),
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        A.Lambda(mask=mask_to_class),
        ToTensorV2()
    ])

def get_3d_augmentation():
    # If you switch to proper TorchIO later
    return Compose([
        # LoadImaged(keys=["image", "mask"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "mask"], source_key="mask"),

    # 2. Random crops, but prefer tumor areas!
        RandCropByPosNegLabeld(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=(512, 512, 60),
            pos=0.9,   # 90% probability to crop tumor (foreground)
            neg=0.1,   # 10% probability to crop background (healthy)
            num_samples=4,  # 4 patches from each volume
        ),

        RandBiasFieldd(keys=["image"], prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.3),
        RandFlipd(keys=["image", "mask"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "mask"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["image", "mask"], spatial_axis=[2], prob=0.5),
        RandZoomd(keys=["image", "mask"], min_zoom=0.9, max_zoom=1.1, prob=0.5),
        mt.Lambda(lambda data: {"mask": mask_to_class(data["mask"]), "image": data["image"]}),

        ToTensord(keys=["image", "mask"]),
    ])








def save_predictions_as_img(
    loader,
    model,
    folder="saved_images/",
    device="cuda",
    max_examples= 20,
):
    os.makedirs(folder, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)    # [B, 3, H, W]
            print("Softmax probs stats:")
            print(f"Min: {probs.min().item():.4f}, Max: {probs.max().item():.4f}, Mean: {probs.mean().item():.4f}")

            # Optionally, print unique probabilities for first pixel (example)
            print("First pixel probs:", probs[0, :, 0, 0])
            preds = probs.argmax(dim=1) 
            print(preds.shape)

            # Move to CPU for saving
            preds = preds.cpu()
            y = y.cpu()
            print("Unique pixel prediction:", np.unique(preds))
            print("Unique pixel mask:", np.unique(y))


            for i in range(x.size(0)):
                if saved >= max_examples:
                    model.train()
                    return

                torchvision.utils.save_image(
                    preds[i].float()/2 ,  # Normalize class indices to [0,1] for viewing (div by num_classes-1)
                    f"{folder}/pred_{saved}.png"
                )
                print(f"pred_{saved}.png")

                # Save ground truth
                torchvision.utils.save_image(
                    y[i].float()/2,        # Same normalization
                    f"{folder}/gt_{saved}.png"
                )
                saved += 1

    model.train()




if __name__ == '__main__':
   pass