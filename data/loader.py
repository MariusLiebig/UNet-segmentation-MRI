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
    def __init__(self, image_paths, mask_paths, augmentation):
        super().__init__(image_paths, mask_paths, augmentation)

    def __getitem__(self, idx):
        img = self.load_nii(self.image_paths[idx])
        mask = self.load_nii(self.mask_paths[idx])
        img, mask = np.expand_dims(img, axis=0), np.expand_dims(mask, axis=0)  # Add channel dimension

        if self.augmentation is not None:
            img, mask = self.apply_augmentations(img, mask)

        # Convert to tensors if not already
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img).float()
        else:
            img = img.clone().detach().float()

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask).float()
        else:
            mask = mask.clone().detach().float()

        return img, mask

    def apply_augmentations(self, image, mask):
        mean = 0.0
        std = 1.0

        # img, msk = crop_resize(image, mask, output_shape=(random.randint(64, 300),random.randint(64, 300), random.randint(32, 52)))
        img, msk = crop_resize(image, mask, output_shape=(128, 128, 32), background_prob=0.2)
    
        # Normalize image
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std

        # Convert to tensors
        msk = mask_to_class(msk)
        # msk = torch.from_numpy(msk).long()
        data = dict(image=img, mask=msk)
        augmented = self.augmentation(data)

        return augmented['image'].unsqueeze(0), augmented['mask']



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
                output_shape=(128, 128, 32),
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
