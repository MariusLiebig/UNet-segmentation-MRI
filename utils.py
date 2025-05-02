import os
# import monai
# from monai.transforms import LoadImage
# import matplotlib.pyplot as plt
from config import base_path
import torch
from data.loader import MedImgDataset2D, MedImgDataset3D
from torch.utils.data import DataLoader, random_split
import torchvision
import glob
from PIL import Image
import numpy as np
import torch.nn.functional as F


def to_cuda(tensor):
    if torch.cuda.is_available():
        if type(tensor) == tuple or type(tensor) == list:
            return [x.cuda() for x in tensor]
        return tensor.cuda()
    return tensor


def data_loader2D(image_paths, mask_paths, train_augmentation, val_augmentation, batch_size, train_set_size = 0.8, keep_background_fraction = 0.1, test = False):
    """
    Load 2D data for training and validation.
    Args:
        image_paths (list): List of paths to the images.
        mask_paths (list): List of paths to the masks.
        train_augmentation (callable): Augmentation function for training data.
        val_augmentation (callable): Augmentation function for validation data.
        batch_size (int): Batch size for DataLoader.
        train_set_size (float): Proportion of data to use for training.
        keep_background_fraction (float): Fraction of background pixels to keep in the dataset.
    """
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

    train_dataset = MedImgDataset2D(train_img_paths, train_mask_paths, augmentation=train_augmentation, keep_background_fraction=keep_background_fraction)
    val_dataset = MedImgDataset2D(val_img_paths, val_mask_paths, augmentation=val_augmentation, keep_background_fraction=1.0)

    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # 3. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def data_loader2D_test(image_paths, mask_paths, augmentation, batch_size, train_set_size = 0.8, keep_background_fraction = 0.1, test = False):
    """
    Load 2D data for training and validation. Only used in testing because I need to reorder the slices
    Args:
        image_paths (list): List of paths to the images.
        mask_paths (list): List of paths to the masks.
        augmentation (callable): Augmentation function for training data.
        batch_size (int): Batch size for DataLoader.
        train_set_size (float): Proportion of data to use for training.
        keep_background_fraction (float): Fraction of background pixels to keep in the dataset.
    """
    #Split into train and validation set via pathdir
    full_dataset = MedImgDataset2D(image_paths, mask_paths, augmentation=augmentation, keep_background_fraction=keep_background_fraction, test = test)
    print(f"Full dataset: {len(full_dataset)}")


    train_size = int(train_set_size * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train dataset length: {len(train_dataset)}, Val dataset length: {len(val_dataset)}")
    # Dataloaders, train_loader -> shuffle = true, val_loader -> shuffle = false
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    
    return train_loader, val_loader



def load_paths():
    """
    Load image and mask paths from the specified directory.
    """

    img_pattern = os.path.join(base_path,'**/*/preRT/*_T2.nii.gz')
    mask_pattern = os.path.join(base_path,'**/*/preRT/*_mask.nii.gz')

    image_paths = glob.glob(img_pattern, recursive=True)
    mask_paths = glob.glob(mask_pattern, recursive=True)

    return image_paths, mask_paths


def save_predictions_as_img(
    loader,
    model,
    epoch, 
    folder="saved_images/",
    device="cuda",
    max_examples= 1,
):
    """
    Save predictions and ground truth images for a given epoch.
    Args:
        loader: DataLoader yielding (images, masks), shape [B, C, H, W]
        model: segmentation model
        epoch: current epoch number
        folder: directory to save image slices
        device: "cuda" or "cpu"
        max_examples: maximum number of examples to save
    """
    os.makedirs(folder, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)    # [B, 3, H, W]
            print("Softmax probs stats:")

            # Optionally, print unique probabilities for first pixel (example)
            print("First pixel probs:", probs[0, :, 0, 0])
            preds = probs.argmax(dim=1) 

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
                    f"{folder}/pred_{saved}_{epoch}.png"
                )
                print(f"pred_{saved}.png")

                # Save ground truth
                torchvision.utils.save_image(
                    y[i].float()/2,        # Same normalization
                    f"{folder}/gt_{saved}_{epoch}.png"
                )
                saved += 1

    model.train()


def mask_to_class(x, **kwargs):
    """
    Convert mask values to class indices.
    Args:
        x: Input mask tensor.
        **kwargs: Additional arguments (not used).
    Returns:
        x_new: Converted mask tensor with class indices.
    """
    x_new = (x == 0.5).astype('uint8') + (x == 1).astype('uint8') * 2

    return x_new




if __name__ == '__main__':
    img_paths, mask_paths = load_paths()
    print(f"Image paths: {img_paths}")
    print(f"Mask paths: {mask_paths}")