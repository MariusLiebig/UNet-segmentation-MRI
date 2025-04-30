import os
import numpy as np
import tensorflow as tf
# from tensorflow.keras.utils import Sequence
import torch
from torch.utils.data import Dataset
# from MiniProject.utils import show_image
import random

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random

#TODO: Add caching, data augmentation, and other preprocessing steps



class BaseDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augmentation=None):
        from monai.transforms import LoadImage
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.loader = LoadImage(image_only=True)
        self.augmentation = augmentation 

    def __len__(self):
        return len(self.image_paths)

    def load_nii(self, path):
        img = self.loader(path)
        img = img.astype(np.float32)
        max_val = np.max(img)
        img /= max_val if max_val > 0 else 1
        return img
    

        
    

class MedImgDataset3D(BaseDataset):
    def __init__(self, image_paths, mask_paths, augmentation, n_crops = 4):
        super().__init__(image_paths, mask_paths, augmentation)
        self.current_volume_idx = -1
        self.current_crops = []
        self.n_crops = n_crops

    def __getitem__(self, idx):
        if self.augmentation is None:
            img = self.load_nii(self.image_paths[idx])
            mask = self.load_nii(self.mask_paths[idx])
            img = np.expand_dims(img, axis=0)    # (1, H, W, D)
            mask = np.expand_dims(mask, axis=0)  # (1, H, W, D)
            return img, mask
        else:
            volume_idx = idx // self.n_crops
            crop_idx = idx % self.n_crops

            # Load and cache new volume + crops if we moved to a new volume
            if volume_idx != self.current_volume_idx:
                img = self.load_nii(self.image_paths[volume_idx])
                mask = self.load_nii(self.mask_paths[volume_idx])
                img = np.expand_dims(img, axis=0)    # (1, H, W, D)
                mask = np.expand_dims(mask, axis=0)  # (1, H, W, D)

                if self.augmentation is not None:
                    self.current_crops = self.apply_augmentations(img, mask)
                else:
                    raise NotImplementedError("You must provide augmentation that returns multiple crops.")

                self.current_volume_idx = volume_idx

            image, mask = self.current_crops[crop_idx]

            assert image.shape[1:] == mask.shape, f"Shape mismatch: image {image.shape}, mask {mask.shape}"
            assert isinstance(image, torch.Tensor), "Image must be a torch.Tensor"
            assert isinstance(mask, torch.Tensor), "Mask must be a torch.Tensor"
            assert image.ndim == 4, f"Image must have 4 dimensions, got {image.ndim}"

            return image, mask

    def __len__(self):
        if self.augmentation is not None:
            return len(self.image_paths) * self.n_crops  # self.n_crops crops per volume
        else:
            return len(self.image_paths)



    def apply_augmentations(self, image, mask):
        mean = 0.0
        std = 1.0

        # Get all crops (mixed tumor and background)
        crops = crop_resize_multiple(image, mask, output_shape=(64, 64, 32), background_fraction=0.2, num_crops=self.n_crops)

        if not crops:
            raise ValueError("No valid crops generated.")

        augmented_crops = []

        for img, msk in crops:
            # Normalize image
            img = img.astype(np.float32) / 255.0
            img = (img - mean) / std

            # Convert mask to class labels
            msk = mask_to_class(msk)

            # Albumentations expects a dict
            data = dict(image=img, mask=msk)
            augmented = self.augmentation(data)

            # Add channel dimension to image → shape becomes (1, H, W, D)
            img_tensor = augmented['image'].unsqueeze(0)
            msk_tensor = augmented['mask']

            augmented_crops.append((img_tensor, msk_tensor))

        return augmented_crops  # List of (image, mask) tuples



class MedImgDataset2D(BaseDataset):
    def __init__(self, image_paths, mask_paths, augmentation = None, slice_axis=2, slice_idx=10, get_all_slices=False, num_slices = 50,
                 keep_background_fraction=0.1):
        super().__init__(image_paths, mask_paths)
        self.slice_axis = slice_axis
        self.slice_idx = slice_idx
        self.augmentation = augmentation

        self.get_all_slices = get_all_slices
        self.num_slices = num_slices

        self.keep_background_fraction = keep_background_fraction
        self.valid_indices = []
        self.prepare_valid_indices()

    def prepare_valid_indices(self):
        """
        Precompute which slices to keep based on presence of cancer.
        """
        print("Preparing dataset slices...")
        for vol_idx in range(len(self.image_paths)):
            mask_volume = self.load_nii(self.mask_paths[vol_idx])

            for slice_idx in range(mask_volume.shape[self.slice_axis]):
                mask_slice = np.take(mask_volume, slice_idx, axis=self.slice_axis)
                
                if np.any(mask_slice > 0):
                    # Cancer exists in slice → always keep
                    self.valid_indices.append((vol_idx, slice_idx))
                else:
                    # No cancer → keep randomly
                    if random.random() < self.keep_background_fraction:
                        self.valid_indices.append((vol_idx, slice_idx))

        print(f"Total kept slices: {len(self.valid_indices)}")
    
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.valid_indices[idx]

        img = self.load_nii(self.image_paths[vol_idx])
        mask = self.load_nii(self.mask_paths[vol_idx])

        img_slice = self.get_slice(img, slice_idx)
        mask_slice = self.get_slice(mask, slice_idx)

        if self.augmentation is not None:
            img_slice, mask_slice = self.apply_augmentations(img_slice, mask_slice)

            # Fix mask shape if necessary
            if mask_slice.ndim == 3 and mask_slice.shape[-1] == 1:
                mask_slice = np.transpose(mask_slice, (2, 0, 1))

        return img_slice, mask_slice

    def get_slice(self, volume, slice_idx):
        slice_2d = np.take(volume, slice_idx, axis=self.slice_axis)
        slice_2d = np.expand_dims(slice_2d, axis=0)  # (C=1, H, W)
        return slice_2d

    def apply_augmentations(self, image, mask):
        image = np.transpose(image, (1, 2, 0))  # (H, W, C)
        mask = np.transpose(mask, (1, 2, 0))
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            return augmented['image'], augmented['mask']
        return image, mask






import numpy as np
import random

def crop_resize(image: np.ndarray,
                mask: np.ndarray,
                output_shape=(32, 32, 32),
                background_prob=0.2):
    """
    Crop a 3D image and mask centered on tumor (non-zero region),
    or randomly sample background with `background_prob`.
    Output shape is fixed as specified (H, W, D).
    """
    image = np.squeeze(image)  # (H, W, D)
    mask = np.squeeze(mask)
    assert image.shape == mask.shape, "Image and mask must have the same shape"

    H, W, D = image.shape
    out_H, out_W, out_D = output_shape

    # Decide whether to do tumor-centered crop or background crop
    force_background = random.random() < background_prob or np.sum(mask) == 0

    if not force_background:
        # --- Tumor-centered crop ---
        coords = np.array(np.nonzero(mask))  # shape (3, N)
        y0, x0, z0 = coords.min(axis=1)
        y1, x1, z1 = coords.max(axis=1)

        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        cz = (z0 + z1) // 2
    else:
        # --- Random background crop ---
        cy = random.randint(out_H // 2, H - out_H // 2)
        cx = random.randint(out_W // 2, W - out_W // 2)
        cz = random.randint(out_D // 2, D - out_D // 2)

    # Compute crop bounds
    y_start = max(cy - out_H // 2, 0)
    x_start = max(cx - out_W // 2, 0)
    z_start = max(cz - out_D // 2, 0)

    y_end = y_start + out_H
    x_end = x_start + out_W
    z_end = z_start + out_D

    # Clamp to valid range
    if y_end > H:
        y_end = H
        y_start = H - out_H
    if x_end > W:
        x_end = W
        x_start = W - out_W
    if z_end > D:
        z_end = D
        z_start = D - out_D

    # Final crop
    img_crop = image[y_start:y_end, x_start:x_end, z_start:z_end]
    msk_crop = mask[y_start:y_end, x_start:x_end, z_start:z_end]

    return img_crop, msk_crop

def mask_to_class(x, **kwargs):
    x_new = (x == 0.5).astype('uint8') + (x == 1).astype('uint8') * 2
    return x_new


def crop_resize_multiple(image: np.ndarray,
                         mask: np.ndarray,
                         output_shape=(32, 32, 32),
                         num_crops=4,
                         background_fraction=0.2,
                         margin_ratio=0.25):
    """
    Returns a mix of tumor-containing and background-only crops.
    - At least (1 - background_fraction) of crops will contain tumor
    - output_shape: (H, W, D)
    """
    image = np.squeeze(image)
    mask = np.squeeze(mask)
    assert image.shape == mask.shape, "Image and mask must have the same shape"

    H, W, D = image.shape
    out_H, out_W, out_D = output_shape

    coords = np.array(np.nonzero(mask))
    if coords.shape[1] == 0:
        raise ValueError("No tumor found in mask.")

    y0, x0, z0 = coords.min(axis=1)
    y1, x1, z1 = coords.max(axis=1)

    # Define tumor region with margin
    margin_y = int(out_H * margin_ratio)
    margin_x = int(out_W * margin_ratio)
    margin_z = int(out_D * margin_ratio)

    y_range = (max(y0 - margin_y, out_H // 2), min(y1 + margin_y, H - out_H // 2))
    x_range = (max(x0 - margin_x, out_W // 2), min(x1 + margin_x, W - out_W // 2))
    z_range = (max(z0 - margin_z, out_D // 2), min(z1 + margin_z, D - out_D // 2))

    crops = []
    num_background = int(num_crops * background_fraction)
    num_tumor = num_crops - num_background

    def extract_crop(cy, cx, cz):
        y_start = max(cy - out_H // 2, 0)
        x_start = max(cx - out_W // 2, 0)
        z_start = max(cz - out_D // 2, 0)

        y_end = y_start + out_H
        x_end = x_start + out_W
        z_end = z_start + out_D

        if y_end > H or x_end > W or z_end > D:
            return None, None

        img_crop = image[y_start:y_end, x_start:x_end, z_start:z_end]
        msk_crop = mask[y_start:y_end, x_start:x_end, z_start:z_end]
        return img_crop, msk_crop

    # --- Tumor crops ---
    attempts = 0
    while len(crops) < num_tumor and attempts < 100:
        cy = random.randint(*y_range)
        cx = random.randint(*x_range)
        cz = random.randint(*z_range)
        img_crop, msk_crop = extract_crop(cy, cx, cz)
        if img_crop is not None and np.any(msk_crop):
            crops.append((img_crop, msk_crop))
        attempts += 1

    # --- Background crops ---
    attempts = 0
    while len(crops) < num_crops and attempts < 100:
        cy = random.randint(out_H // 2, H - out_H // 2)
        cx = random.randint(out_W // 2, W - out_W // 2)
        cz = random.randint(out_D // 2, D - out_D // 2)
        img_crop, msk_crop = extract_crop(cy, cx, cz)
        if img_crop is not None and not np.any(msk_crop):
            crops.append((img_crop, msk_crop))
        attempts += 1

    if len(crops) < num_crops:
        print(f"Warning: only {len(crops)} valid crops were generated.")

    return crops



if __name__ == "__main__":
    base_path = "/home/mariusliebig/Documents/DYP/HNTS-MRG"
    img_path = [
                "HNTSMRG24_train/10/midRT/10_midRT_T2.nii.gz", 
                "HNTSMRG24_train/8/midRT/8_midRT_T2.nii.gz",
                "HNTSMRG24_train/6/midRT/6_midRT_T2.nii.gz",
                "HNTSMRG24_train/4/midRT/4_midRT_T2.nii.gz",
                "HNTSMRG24_train/2/midRT/2_midRT_T2.nii.gz"
                ]
    m_path = [
            "HNTSMRG24_train/10/midRT/10_midRT_mask.nii.gz", 
            "HNTSMRG24_train/8/midRT/8_midRT_mask.nii.gz",
            "HNTSMRG24_train/6/midRT/6_midRT_mask.nii.gz",
            "HNTSMRG24_train/4/midRT/4_midRT_mask.nii.gz",
            "HNTSMRG24_train/2/midRT/2_midRT_mask.nii.gz"
            ]
    image_paths = [os.path.join(base_path, path) for path in img_path ]
    mask_paths = [os.path.join(base_path, path) for path in m_path]

    dataset = MedImgDataset2D(image_paths, mask_paths)


    for X, y in dataset:
        print(f"  Image shape: {X.shape}")
        print(f"  Mask shape: {y.shape}")
        # show_image(X)
