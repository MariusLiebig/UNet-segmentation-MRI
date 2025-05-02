import os
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset
import random
import sys
from monai.transforms import LoadImage




class BaseDataset(Dataset):
    """
    Base dataset for loading 2D and 3D medical image and masks.
    Args:
        image_paths (list): List of paths to the images.
        mask_paths (list): List of paths to the masks.
        augmentation (callable, optional): Optional augmentation function to be applied on the images and masks.
    """
    def __init__(self, image_paths, mask_paths, augmentation=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.loader = LoadImage(image_only=True)
        self.augmentation = augmentation 
        self.img_volumes = [self.load_nii(p) for p in image_paths]
        self.mask_volumes = [self.load_nii(p) for p in mask_paths]

    def __len__(self):
        return len(self.image_paths)

    def load_nii(self, path):
        """
        Load a NIfTI file.
        Args:
            path (str): Path to the NIfTI file.
        """
        img = self.loader(path)
        img = img.astype(np.float32)
        max_val = np.max(img)
        return img
    
    def apply_augmentations(self, image, mask):
        """
        Apply augmentations to the image and mask 
        Args:
            image (numpy.ndarray): Image to be augmented. Shape: (C, H, W)
            mask (numpy.ndarray): Mask to be augmented. Shape: (C, H, W)
        """
        image = np.transpose(image, (1, 2, 0))  # (H, W, C)
        mask = np.transpose(mask, (1, 2, 0)) 
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            return augmented['image'], augmented['mask']
        return image, mask
    

class MedImgDataset3D(BaseDataset):
    """
    3D Dataset for full volume inference or training.
    Args:
        image_paths (list): List of paths to the images.
        mask_paths (list): List of paths to the masks.
        augmentation (callable, optional): Optional augmentation function to be applied on the images and masks.
    """
    def __init__(self, image_paths, mask_paths, augmentation=None):
        super().__init__(image_paths, mask_paths, augmentation)

    def __getitem__(self, idx):
        """
        Returns a full 3D image volume and its mask.
        """
        img = self.load_nii(self.image_paths[idx])
        mask = self.load_nii(self.mask_paths[idx])
        img = torch.tensor(img).float()
        mask = torch.tensor(mask).float()
        img, mask = self.apply_augmentations(img, mask)

        img = self.pad_volume(img)

        return torch.tensor(img).float(), torch.tensor(mask).float()





class MedImgDataset2D(BaseDataset):
    """
    2D Slice Dataset for training or evaluation.
    Args:
        image_paths (list): List of paths to the images.
        mask_paths (list): List of paths to the masks.
        augmentation (callable, optional): Optional augmentation function to be applied on the images and masks.
        slice_axis (int): Axis along which to slice the 3D volume. Default is 2 (slicing along depth).
        keep_background_fraction (float): Fraction of background slices to keep. Default is 0.05.
    """
    def __init__(self, image_paths, mask_paths, augmentation = None, slice_axis=2, keep_background_fraction=0.05, test = False):
        super().__init__(image_paths, mask_paths)
        self.slice_axis = slice_axis
        self.augmentation = augmentation

        self.keep_background_fraction = keep_background_fraction
        self.valid_indices = []
        self.prepare_valid_indices()

        self.vol_idx = [self.valid_indices[i][0] for i in range(len(self.valid_indices))]
        self.slice_idx = [self.valid_indices[i][1] for i in range(len(self.valid_indices))]

        self.img_slice = [self.get_slice(self.img_volumes[self.vol_idx[i]], self.slice_idx[i]) for i in range(len(self.valid_indices))]
        self.mask_slice = [self.get_slice(self.mask_volumes[self.vol_idx[i]], self.slice_idx[i]) for i in range(len(self.valid_indices))]

        self.test = test


    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Returns a single slice (and optionally its index).
        Idx is used when reconstructing the full volume while testing.
        """
        if self.augmentation is not None:
            img_slice, mask_slice = self.apply_augmentations(self.img_slice[idx], self.mask_slice[idx])

            # Fix mask shape if necessary
            if mask_slice.ndim == 3 and mask_slice.shape[-1] == 1:
                mask_slice = np.transpose(mask_slice, (2, 0, 1))
        if self.test:
            return img_slice, mask_slice, idx
        else:
            return img_slice, mask_slice

    

    def prepare_valid_indices(self):
        """
        Precompute and store (volume_idx, slice_idx) tuples to include in dataset.
        Always include slices with cancer; include some background-only slices randomly.
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
    
    def get_slice(self, volume, slice_idx):
        """
        Extracts a 2D slice from a 3D volume along the specified axis.
        Args:
            volume (numpy.ndarray): 3D volume from which to extract the slice.
            slice_idx (int): Index of the slice to extract.
        Returns shape: (C=1, H, W)
        """
        slice_2d = np.take(volume, slice_idx, axis=self.slice_axis)
        slice_2d = np.expand_dims(slice_2d, axis=0)  # (C=1, H, W)
        return slice_2d



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
