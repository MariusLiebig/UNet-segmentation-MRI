import torch
import nibabel as nib
import numpy as np
import os
from config import base_path
from model import UNET3D
from config import CONFIG

def load_model(checkpoint_path, device="cuda"):
    # Create model architecture
    model = UNET().to(device)

    # Load saved weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    return model


def load_model3D(checkpoint_path, device="cuda"):
    # Create model architecture
    model = UNET3D(feature_size = CONFIG["feature_sizes"]).to(device)

    # Load saved weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    return model


def predict_and_save_all_slices(model, input_path, save_path, device="cuda", slice_axis=2):
    # Load 3D input NIfTI image
    nii = nib.load(input_path)
    volume = nii.get_fdata()
    affine = nii.affine
    header = nii.header
    original_shape = volume.shape

    num_slices = original_shape[slice_axis]

    pred_slices = []

    model.eval()
    with torch.no_grad():
        for slice_idx in range(num_slices):
            slice_2d = get_slice(volume, slice_idx=slice_idx, slice_axis=slice_axis)

            img = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0).to(device)  # (B, C, H, W)

            logits = model(img)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)  # (B, H, W)

            preds = preds.squeeze(0).cpu().numpy()  # (H, W)

            pred_slices.append(preds)

    # Now stack correctly
    pred_volume = np.stack(pred_slices, axis=0)  # Always (D, H, W)

    # If slice_axis is not 0, rearrange dimensions
    if slice_axis == 0:
        pred_volume = pred_volume  # (D, H, W) -> (D, H, W)
    elif slice_axis == 1:
        pred_volume = np.transpose(pred_volume, (1, 0, 2))  # (H, D, W)
    elif slice_axis == 2:
        pred_volume = np.transpose(pred_volume, (1, 2, 0))  # (H, W, D)

    print(f"Final predicted volume shape: {pred_volume.shape}")

    # Save output volume as NIfTI
    pred_img = nib.Nifti1Image(pred_volume.astype(np.uint8), affine=affine, header=header)
    nib.save(pred_img, save_path)
    print(f"Saved 3D prediction to {save_path}")

def get_slice(volume, slice_idx, slice_axis=2):
    slice_2d = np.take(volume, slice_idx, axis=slice_axis)
    slice_2d = np.expand_dims(slice_2d, axis=0)  # (C=1, H, W)
    return slice_2d



def predict_and_save_volume(model, input_path, save_path, device="cuda"):
    # Load full 3D NIfTI image
    nii = nib.load(input_path)
    volume = nii.get_fdata()
    affine = nii.affine
    header = nii.header

    # Prepare input
    if not isinstance(volume, torch.Tensor):
        img = torch.tensor(volume, dtype=torch.float32)
    else:
        img = volume.clone().detach().float()

    # Add batch and channel dimensions if missing
    if img.ndim == 3:
        img = img.unsqueeze(0).unsqueeze(0)  # (B=1, C=1, H, W, D)
    elif img.ndim == 4:
        img = img.unsqueeze(0)  # (B=1, C, H, W, D)

    img = img.to(device)

    # Predict full volume
    model.eval()
    with torch.no_grad():
        logits = model(img)  # (B=1, C, H, W, D)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)  # (B=1, H, W, D)

    # Remove batch dimension
    preds = preds.squeeze(0).cpu().numpy()  # (H, W, D)

    print(f"Predicted volume shape: {preds.shape}")

    # Save as NIfTI
    pred_img = nib.Nifti1Image(preds.astype(np.uint8), affine=affine, header=header)
    nib.save(pred_img, save_path)

    print(f"Saved full 3D prediction to {save_path}")


if __name__ == "__main__":
    checkpoint_path = "checkpoints/checkpoint_epoch_10.pth"
    img_path = "train/10/preRT/10_preRT_T2.nii.gz"
            
    image_paths = os.path.join(base_path, img_path) 
    save_path = "predictions/pred_new_image.nii.gz"

    os.makedirs("predictions", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model3D(checkpoint_path, device=device)

    # Predict and save
    predict_and_save_volume(model, image_paths, save_path, device=device)
