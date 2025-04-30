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
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandAffined,
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def data_loader3D(image_paths, mask_paths, augmentation, batch_size, train_set_size = 0.8):
    total_size = len(image_paths)
    train_size = int(train_set_size * total_size)
    val_size = total_size - train_size

    # Random split of indices
    indices = list(range(total_size))
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_img_paths = [image_paths[i] for i in train_indices]
    train_mask_paths = [mask_paths[i] for i in train_indices]
    val_img_paths = [image_paths[i] for i in val_indices]
    val_mask_paths = [mask_paths[i] for i in val_indices]

    print(f"Train image paths: {len(train_img_paths)}")
    print(f"Train mask paths: {len(train_mask_paths)}")
    print(f"Validation image paths: {len(val_img_paths)}")
    print(f"Validation mask paths: {len(val_mask_paths)}")


    train_dataset = MedImgDataset3D(train_img_paths, train_mask_paths, augmentation=augmentation)
    val_dataset = MedImgDataset3D(val_img_paths, val_mask_paths, augmentation=None)  # usually no augmentation for validation

    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # 3. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader

def load_paths():

    img_pattern = os.path.join(base_path,'**/*/preRT/*_T2.nii.gz')
    mask_pattern = os.path.join(base_path,'**/*/preRT/*_mask.nii.gz')

    image_paths = glob.glob(img_pattern, recursive=True)
    mask_paths = glob.glob(mask_pattern, recursive=True)

    return image_paths, mask_paths


def mask_to_class(x, **kwargs):
    #   raw x = [0, ~0.498, 1.0]
    c1 = np.isclose(x, 127/255, atol=1e-2)   # everything near 0.498 → class 1
    c2 = np.isclose(x,   1.0,     atol=1e-6) # exactly normalized 255 → class 2
    return (c1.astype('uint8') + 2*c2.astype('uint8'))




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
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 2.0), mode=("bilinear", "nearest")),
# Aktiviere diese Transformationen
        RandFlipd(keys=["image", "mask"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "mask"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["image", "mask"], spatial_axis=[2], prob=0.5),
        
        # Intensitätstransformationen verstärken
        RandBiasFieldd(keys=["image"], prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.2, prob=0.7),
        RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
        
        # Zusätzliche Transformationen
        RandAdjustContrastd(keys=["image"], prob=0.3),
        RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0)),
        
        # Elastische Deformation für realistische Variationen
        RandAffined(
            keys=["image", "mask"],
            prob=0.3,
            rotate_range=(0.05, 0.05, 0.05),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
        ),
        RandZoomd(keys=["image", "mask"], min_zoom=0.9, max_zoom=1.1, prob=0.5),
        # mt.Lambda(lambda data: {"mask": mask_to_class(data["mask"]), "image": data["image"]}),

        ToTensord(keys=["image", "mask"]),
    ])




def compute_class_frequencies(train_loader, num_classes=3):
    voxel_counts = np.zeros(num_classes, dtype=np.int64)
    print(f"Length of train_loader: {len(train_loader)}")
    for _, masks in train_loader:
        # Ensure masks are on CPU and in integer format
        masks = masks.long().cpu()

        for cls in range(num_classes):
            voxel_counts[cls] += torch.sum(masks == cls).item()
        print(f"Voxel counts for class {cls}: {voxel_counts}")

    return voxel_counts




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