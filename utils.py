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

def data_loader3D(image_paths, mask_paths, augmentation, batch_size, train_set_size = 0.8):
    full_dataset = MedImgDataset3D(image_paths, mask_paths, augmentation=augmentation)
    print(f"Full dataset length: {len(full_dataset)}")

    train_size = int(train_set_size * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Randomly split dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Dataloaders, train_loader -> shuffle = true, val_loader -> shuffle = false
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def data_loader2D_test(image_paths, mask_paths, augmentation, batch_size, train_set_size = 0.8, keep_background_fraction = 0.1, test = False):
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

    img_pattern = os.path.join(base_path,'**/*/preRT/*_T2.nii.gz')
    mask_pattern = os.path.join(base_path,'**/*/preRT/*_mask.nii.gz')

    image_paths = glob.glob(img_pattern, recursive=True)
    mask_paths = glob.glob(mask_pattern, recursive=True)

    return image_paths, mask_paths




def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])



def save_predictions_as_img(
    loader,
    model,
    epoch, 
    folder="saved_images/",
    device="cuda",
    max_examples= 1,
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



def save_predictions_as_img_3d(
    loader,
    model,
    folder="saved_images/",
    device="cuda",
    overlay: bool = False,
    slice_axis: int = 2,  # Default: axial slices
    save_every_nth_slice: int = 1,  # Save every slice, or every nth
):
    """
    Save predictions and overlays for 3D volumes slice-by-slice.

    Args:
        loader: DataLoader yielding (images, masks), shape [B, C, D, H, W]
        model: segmentation model
        folder: directory to save image slices
        device: "cuda" or "cpu"
        overlay: if True, save blended overlay image
        slice_axis: which axis to slice on (0=depth, 1=height, 2=width)
        save_every_nth_slice: skip slices for brevity if desired
    """
    os.makedirs(folder, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(device)
            preds = torch.sigmoid(model(imgs))
            preds = (preds > 0.5).float().cpu().numpy()  # [B, 1, D, H, W]
            imgs_np = imgs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(imgs_np.shape[0]):  # batch loop
                idx = batch_idx * loader.batch_size + i
                pred_vol = preds[i, 0]   # [D, H, W]
                img_vol = imgs_np[i, 0]  # [D, H, W] or [C, D, H, W]
                gt_vol = masks_np[i, 0]  # [D, H, W]

                depth = pred_vol.shape[0]

                for d in range(0, depth, save_every_nth_slice):
                    pred_slice = (pred_vol[d] * 255).astype(np.uint8)
                    gt_slice = (gt_vol[d] * 255).astype(np.uint8)
                    img_slice = (img_vol[d] * 255).astype(np.uint8)

                    Image.fromarray(pred_slice).save(f"{folder}/pred_{idx}_{d}.png")
                    Image.fromarray(gt_slice).save(f"{folder}/gt_{idx}_{d}.png")

                    if overlay:
                        overlay_img = np.stack([img_slice] * 3, axis=-1)  # grayscale to RGB
                        overlay_img[pred_slice > 0] = [255, 0, 0]  # red mask
                        blended = (0.6 * overlay_img + 0.4 * np.stack([img_slice]*3, axis=-1)).astype(np.uint8)
                        Image.fromarray(blended).save(f"{folder}/overlay_{idx}_{d}.png")

    model.train()

def mask_to_class(x, **kwargs):
    x_new = (x == 0.5).astype('uint8') + (x == 1).astype('uint8') * 2

    return x_new




# def path_show_image(img_path, every_nth=5, img_only=True, channel_first=True, simp_keys=True):
#     data = LoadImage(image_only = img_only, ensure_channel_first = channel_first, simple_keys = simp_keys)(os.path.join(base_path, img_path))
#     print(f"image data shape: {data.shape}")
#     print(f"meta data: {data.meta.keys()}")
#     fig, _ = monai.visualize.matshow3d(monai.transforms.Orientation("SPL")(data), every_n = every_nth)
#     plt.show()

# def show_image(image, every_nth=5):
#     fig, _ = monai.visualize.matshow3d(monai.transforms.Orientation("SPL")(image), every_n = every_nth)
#     plt.show()




if __name__ == '__main__':
    img_paths, mask_paths = load_paths()
    print(f"Image paths: {img_paths[0:5]}")
    dataset = MedImgDataset3D(img_paths, mask_paths)

    largest_height = 0
    largest_width = 0
    largest_breadth = 0
    mask_largest_height = 0
    mask_largest_width = 0
    mask_largest_breadth = 0

    for X, y in dataset:
        largest_height = max(largest_height, X.shape[0])
        largest_width = max(largest_width, X.shape[1])
        largest_breadth = max(largest_breadth, X.shape[2])
        mask_largest_height = max(mask_largest_height, y.shape[0])
        mask_largest_width = max(mask_largest_width, y.shape[1])
        mask_largest_breadth = max(mask_largest_breadth, y.shape[2])
        print(f"  Image shape: {X.shape}")
        print(f"  Mask shape: {y.shape}")
        # show_image(X)
    print(f"Largest height: {largest_height}")
    print(f"Largest width: {largest_width}")
    print(f"Largest breadth: {largest_breadth}")
    print(f"Mask largest height: {mask_largest_height}")
    print(f"Mask largest width: {mask_largest_width}")
    print(f"Mask Largest breadth: {mask_largest_breadth}")
