import torch
import nibabel as nib
import numpy as np
import os
from config import base_path
from model import UNET
from utils import data_loader2D, mask_to_class, save_predictions_as_img, data_loader2D_test
import albumentations as A
from albumentations.pytorch import ToTensorV2
from metric import dice_coefficient


def load_model(checkpoint_path, device="cuda"):
    # Create model architecture
    model = UNET().to(device)

    # Load saved weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    return model




def predict_and_save_all_slices(model, input_path, mask_path, save_path, number, device="cuda", slice_axis=2):
    # Load 3D input NIfTI image
    #---------Just to get the affine and header ------------------
    nii = nib.load(input_path)
    volume = nii.get_fdata()
    affine = nii.affine
    header = nii.header
    #---------------- Not needed for prediction ---------------------

    image_height = 512
    image_width = 512
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

    data, _ = data_loader2D([input_path], [mask_path], augmentation, 1, 1, keep_background_fraction = 1.0)
    dice, _ = dice_coefficient(data, model)
    print(f"Dice coefficient: {dice}")
    data, _ = data_loader2D_test([input_path], [mask_path], augmentation, 1, 1, keep_background_fraction = 1.0, test=True)




    # dice, _ = dice_coefficient(data, model)
    # print(f"Dice coefficient: {dice}")

    # reordered_data = [None] * len(idx)

    # # Put each item from `my_list` into its *new* position
    # for new_idx, old_idx in enumerate(idx):
    #     print(f"New index: {new_idx}, Old index: {old_idx}")
    #     reordered_data[new_idx] = data[old_idx]

    # data = reordered_img
    # mask = reordered_mask

    pred_slices = []
    mask_slices = []
    indexes = []




    model.eval()
    with torch.no_grad():
        for slice_2d, mask, idx in data:

            img = torch.tensor(slice_2d, dtype=torch.float32).to(device)  # (B, C, H, W)

            logits = model(img)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)  # (B, H, W)
            # Move to CPU for saving

            preds = preds.squeeze(0).cpu().numpy()  # (H, W)
            mask = mask.squeeze(0)  # (H, W)
            mask = mask.squeeze(0).cpu().numpy()  # (H, W)

            print(f"Preds shape: {preds.shape}")
            print(f"Mask shape: {mask.shape}")
            # Convert to 2D slice
            pred_slices.append(preds)
            mask_slices.append(mask)
            indexes.append(idx)

    reordered_mask = [None] * len(indexes)
    reordered_img = [None] * len(indexes)

    # Put each item from `my_list` into its *new* position
    for new_idx, old_idx in enumerate(indexes):
        old_idx = old_idx.item() if isinstance(old_idx, torch.Tensor) else old_idx
        print(f"New index: {new_idx}, Old index: {old_idx}, Slice: {len(mask_slices)}")
        reordered_mask[old_idx] = mask_slices[new_idx]
        reordered_img[old_idx] = pred_slices[new_idx]

    pred_slices = reordered_img
    mask_slices = reordered_mask

    # Now stack correctly
    pred_volume = np.stack(pred_slices, axis=0)  # Always (D, H, W)
    mask_volume = np.stack(mask_slices, axis=0)  # (D, H, W)

    print(f"Predicted volume shape: {pred_volume.shape}")
    print(f"Mask volume shape: {mask_volume.shape}")

    # If slice_axis is not 0, rearrange dimensions
    if slice_axis == 0:
        pred_volume = pred_volume  # (D, H, W) -> (D, H, W)
        mask_volume = mask_volume  # (D, H, W) -> (D, H, W)
    elif slice_axis == 1:
        pred_volume = np.transpose(pred_volume, (1, 0, 2))  # (H, D, W)
        mask_volume = np.transpose(mask_volume, (1, 0, 2))
    elif slice_axis == 2:
        pred_volume = np.transpose(pred_volume, (1, 2, 0))  # (H, W, D)
        mask_volume = np.transpose(mask_volume, (1, 2, 0))

    print(f"Final predicted volume shape: {pred_volume.shape}")

    # Save output volume as NIfTI
    pred_img = nib.Nifti1Image(pred_volume.astype(np.uint8), affine=affine, header=header)
    mask_img = nib.Nifti1Image(mask_volume.astype(np.uint8), affine=affine, header=header)
    nib.save(pred_img, save_path + f"_{number}_3_2.nii.gz")
    nib.save(mask_img, save_path + f"_{number}_mask_2.nii.gz")
    print(f"Saved prediction to {save_path}_{number}_3.nii.gz")
    print(f"Saved 3D prediction to {save_path}")

def get_slice(volume, slice_idx, slice_axis=2):
    slice_2d = np.take(volume, slice_idx, axis=slice_axis)
    slice_2d = np.expand_dims(slice_2d, axis=0)  # (C=1, H, W)
    return slice_2d



if __name__ == "__main__":
    checkpoint_path = "checkpoints/checkpoint_epoch_13.pth"
    number = 10
    img_path = f"train/{number}/preRT/{number}_preRT_T2.nii.gz"
    mask_path = f"train/{number}/preRT/{number}_preRT_mask.nii.gz"
            
    image_paths = os.path.join(base_path, img_path) 
    mask_paths = os.path.join(base_path, mask_path)
    save_path = "predictions/pred_new_image"

    os.makedirs("predictions", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model(checkpoint_path, device=device)

    # Predict and save
    predict_and_save_all_slices(model, image_paths, mask_paths, save_path, number, device=device)
