from config import base_path
from data.loader import MedImgDataset3D, MedImgDataset2D
import os
import torch.nn.functional as F
import torch
import torch.nn as nn

from utils import to_cuda

def dice_coefficient(loader, model, loss_fn = None, num_classes=3, device="cuda"):
    """
    Computes average Dice score over a DataLoader for multi-class segmentation.

    Args:
        loader      : DataLoader yielding (imgs, masks), where masks have values in {0..num_classes-1}
        model       : Model outputting logits of shape (B, num_classes, H, W)
        loss_fn     : Optional loss function to compute loss
        num_classes : Number of segmentation classes
        device      : 'cuda' or 'cpu'
    """
    model.eval()
    total_dice = 0.0
    n_batches = 0
    total_loss = 0.0
    eps = 1e-6
    total_intersection = torch.zeros(num_classes, device=device)
    total_cardinality = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)  # convert (B, 1, H, W) → (B, H, W)

            masks = masks.long()  # Ensure class indices

            # Forward
            logits = model(imgs)  # (B, C, H, W)
            preds = logits.argmax(dim=1)  # (B, H, W)

            # One-hot encode to (B, C, H, W)
            masks_onehot = F.one_hot(masks, num_classes=num_classes)  # (B, H, W, C)
            masks_onehot = masks_onehot.permute(0, 3, 1, 2).float()    # → (B, C, H, W)

            preds_onehot = F.one_hot(preds, num_classes=num_classes)  # (B, H, W, C)
            preds_onehot = preds_onehot.permute(0, 3, 1, 2).float()

            # Dice computation
            intersection = (preds_onehot * masks_onehot).sum(dim=(2, 3))  # (B, C)
            cardinality = preds_onehot.sum(dim=(2, 3)) + masks_onehot.sum(dim=(2, 3))  # (B, C)

            dice_per_class = (2. * intersection + eps) / (cardinality + eps)  # (B, C)
            dice_per_sample = dice_per_class.mean(dim=1)  # average over classes → (B,)
            dice_batch = dice_per_sample.mean().item()

            total_dice += dice_batch
            n_batches += 1

            if loss_fn is not None:
                loss = loss_fn(logits, masks)
                total_loss += loss.item()

    dice_per_class_avg = ((2. * total_intersection + eps) / (total_cardinality + eps)).tolist()
    avg_dice = total_dice / max(1, n_batches)
    avg_loss = total_loss / max(1, n_batches) if loss_fn is not None else None

    model.train()
    return avg_dice, avg_loss, dice_per_class_avg

class DiceLoss(nn.Module):
    def __init__(self, num_classes, weights, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.weights = weights

    def forward(self, logits, targets):
        # logits: [B, C, H, W], targets: [B, H, W]
        B, C, H, W = logits.shape
        assert C == self.num_classes, f"Expected {self.num_classes} classes, got {C} in logits"

        # One-hot encode targets to shape (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()
        
        # Apply softmax to logits
        probs = F.softmax(logits, dim=1).clamp(min=1e-6, max=1 - 1e-6)

        if probs.shape != targets_one_hot.shape:
            raise ValueError(f"Shape mismatch: probs {probs.shape} vs targets_one_hot {targets_one_hot.shape}")

        # Flatten spatial dims
        probs_flat = probs.view(B, C, -1)
        targets_flat = targets_one_hot.view(B, C, -1)

        # Dice computation
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        class_weights = self.weights
        dice = (1 - dice) * class_weights
        selected = class_weights > 0
        loss = dice[:, selected].mean()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [B, C, H, W]
        # targets: [B, H, W]

        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # [B, H, W]
        pt = torch.exp(-ce_loss)  # probability of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, weights, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss(num_classes, weights)
        self.weights = weights
        self.ce = FocalLoss(alpha=1, gamma=2, reduction="mean")
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        dice_loss = self.dice(logits, targets)
        ce_loss = self.ce(logits, targets)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss



if __name__ == "__main__":
    print("Loading dataset...")
    img_path = [
                "HNTSMRG24_train/10/midRT/10_preRT_mask_registered.nii.gz", 
                "HNTSMRG24_train/8/midRT/8_preRT_mask_registered.nii.gz",
                "HNTSMRG24_train/6/midRT/6_preRT_mask_registered.nii.gz",
                "HNTSMRG24_train/4/midRT/4_preRT_mask_registered.nii.gz",
                "HNTSMRG24_train/2/midRT/2_preRT_mask_registered.nii.gz"
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

    dataset = MedImgDataset3D(
        image_paths = image_paths,
        mask_paths = mask_paths,

    )
