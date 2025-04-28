from config import base_path
from data.loader import MedImgDataset3D, MedImgDataset2D
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from monai.networks.utils import one_hot

from utils import to_cuda

import torch.nn.functional as F

from monai.networks.utils import one_hot

def dice_coefficient(loader, model, loss_fn=None, num_classes=3, device="cuda"):
    model.eval()
    total_dice = 0.0
    total_loss = 0.0
    n_batches = 0
    eps = 1e-6

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            elif masks.ndim == 5 and masks.shape[1] == 1: 
                masks = masks.squeeze(1)
                

            masks = masks.long()

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            if loss_fn is not None:
                loss = loss_fn(logits, masks)
                total_loss += loss.item()

            one_hot_dims = tuple(list([0, masks.ndim]) + list(range(1, masks.ndim))) #For 2D (0, 3, 2, 1) and for 3D (0, 4, 3, 2, 1)
            masks_onehot = F.one_hot(masks, num_classes=num_classes).permute(one_hot_dims).float()
            preds_onehot = F.one_hot(preds, num_classes=num_classes).permute(one_hot_dims).float()

            sum_dims = tuple(range(2, preds_onehot.ndim))  
            intersection = (preds_onehot * masks_onehot).sum(dim=sum_dims)
            cardinality = preds_onehot.sum(dim=sum_dims) + masks_onehot.sum(dim=sum_dims)

            dice_per_class = (2. * intersection + eps) / (cardinality + eps)
            dice_per_sample = dice_per_class.mean(dim=1)
            dice_batch = dice_per_sample.mean().item()

            total_dice += dice_batch
            n_batches += 1

    avg_dice = total_dice / max(1, n_batches)
    avg_loss = total_loss / max(1, n_batches) if loss_fn is not None else None

    # print(f"Average Dice over {n_batches} batches: {avg_dice:.4f}")
    if avg_loss is not None:
        print(f"Average Validation Loss: {avg_loss:.4f}")

    model.train()

    return avg_dice, avg_loss





from monai.networks.utils import one_hot

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = torch.tensor([0.05, 0.475, 0.475], device='cuda')  # move to init

    def forward(self, logits, targets):
        if logits.ndim == 5 :
            B, C, H, W, D = logits.shape
        else:
            B, C, H, W = logits.shape
            
        assert C == self.num_classes, f"Expected {self.num_classes} classes, got {C} in logits"
        one_hot_dims = tuple(list([0, targets.ndim]) + list(range(1, targets.ndim))) #For 2D (0, 3, 2, 1) and for 3D (0, 4, 3, 2, 1)
        targets_one_hot = F.one_hot(targets, num_classes=C).permute(one_hot_dims).float()

        probs = F.softmax(logits, dim=1)

        if probs.shape != targets_one_hot.shape:
            raise ValueError(f"Shape mismatch: probs {probs.shape} vs targets_one_hot {targets_one_hot.shape}")
        probs_flat = probs.view(B, C, -1)
        targets_flat = targets_one_hot.view(B, C, -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        dice = (1 - dice) * self.class_weights
        dice = (1 - dice) 
        loss = dice.mean()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [B, C, H, W, D]
        # targets: [B, H, W, D]

        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # [B, H, W, D]
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, dice_weight=0.7, ce_weight=0.3):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.weights = torch.tensor([0.2, 0.4, 0.4], device='cuda')
        self.alpha = torch.tensor([0.05, 0.475, 0.475], device='cuda')

        self.ce = FocalLoss(alpha=self.alpha, gamma=2, reduction="mean")
        # self.ce = nn.CrossEntropyLoss(weight=self.weights, reduction="mean")
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
