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
import gc
from monai.inferers import SlidingWindowInferer


def dice_coefficient(loader, model, inferer, loss_fn=None, num_classes=3, device="cuda"):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()

    total_dice = 0.0
    total_loss = 0.0
    n_batches = 0
    eps = 1e-6

    total_intersection = torch.zeros(num_classes, device=device)
    total_cardinality = torch.zeros(num_classes, device=device)



    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

            elif masks.ndim == 5 and masks.shape[1] == 1: 
                masks = masks.squeeze(1)
                

            masks = masks.long()
            logits = inferer(inputs=imgs, network=model)
            preds = logits.argmax(dim=1)

            if loss_fn is not None:
                loss = loss_fn(logits, masks)
                total_loss += loss.item()

            one_hot_dims = (0, masks.ndim) + tuple(range(1, masks.ndim))
            masks_onehot = F.one_hot(masks, num_classes=num_classes).permute(one_hot_dims).float()
            preds_onehot = F.one_hot(preds, num_classes=num_classes).permute(one_hot_dims).float()

            sum_dims = tuple(range(2, preds_onehot.ndim))  
            intersection = (preds_onehot * masks_onehot).sum(dim=sum_dims)  # (B, C)
            cardinality = preds_onehot.sum(dim=sum_dims) + masks_onehot.sum(dim=sum_dims)  # (B, C)

            # Accumulate per class
            total_intersection += intersection.sum(dim=0)
            total_cardinality += cardinality.sum(dim=0)

            dice_per_class = (2. * intersection + eps) / (cardinality + eps)  # (B, C)
            dice_per_sample = dice_per_class.mean(dim=1)
            dice_batch = dice_per_sample.mean().item()

            total_dice += dice_batch
            n_batches += 1

            del imgs, masks, logits, preds, masks_onehot, preds_onehot
            torch.cuda.empty_cache()

    avg_dice = total_dice / max(1, n_batches)
    dice_per_class_avg = (2. * total_intersection + eps) / (total_cardinality + eps)

    avg_loss = total_loss / max(1, n_batches) if loss_fn is not None else None

    model.train()
    gc.collect()
    torch.cuda.empty_cache()

    return avg_dice, avg_loss, dice_per_class_avg.cpu().tolist()





from monai.networks.utils import one_hot

class DiceLoss(nn.Module):
    def __init__(self, num_classes, weights, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = torch.as_tensor(weights, dtype=torch.float32)
        if not self.class_weights.is_cuda:
            self.class_weights = self.class_weights.to('cuda')

    def forward(self, logits, targets):
        if logits.ndim == 5 :
            B, C, H, W, D = logits.shape
        else:
            B, C, H, W = logits.shape
            
        assert C == self.num_classes, f"Expected {self.num_classes} classes, got {C} in logits"
        one_hot_dims = tuple(list([0, targets.ndim]) + list(range(1, targets.ndim))) #For 2D (0, 3, 2, 1) and for 3D (0, 4, 3, 2, 1)
        targets_one_hot = F.one_hot(targets, num_classes=C).permute(one_hot_dims).float()

        probs = F.softmax(logits, dim=1)
        probs = probs.clamp(min=1e-6, max=1.0 - 1e-6) #so they stay in a safe numerical range


        if probs.shape != targets_one_hot.shape:
            raise ValueError(f"Shape mismatch: probs {probs.shape} vs targets_one_hot {targets_one_hot.shape}")
        probs_flat = probs.view(B, C, -1)
        targets_flat = targets_one_hot.view(B, C, -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        dice = (1 - dice) * self.class_weights
        loss = dice.mean()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Expect a tensor or float
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # shape: (B, H, W, D)
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            # Ensure alpha is a tensor on correct device
            if not isinstance(self.alpha, torch.Tensor):
                alpha = torch.tensor(self.alpha, dtype=torch.float32, device=logits.device)
            else:
                alpha = self.alpha.to(logits.device)

            # Reshape for broadcasting: (1, C, 1, 1, 1)
            alpha = alpha.view(1, -1, 1, 1, 1)

            # Convert targets to one-hot: (B, C, H, W, D)
            num_classes = logits.shape[1]
            targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

            # Compute per-voxel alpha
            alpha_t = (alpha * targets_onehot).sum(dim=1)  # shape: (B, H, W, D)

            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, class_weights, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss(num_classes, weights=class_weights)
        # self.ce = FocalLoss(alpha=class_weights, gamma=2, reduction="mean")
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to("cuda"))
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
