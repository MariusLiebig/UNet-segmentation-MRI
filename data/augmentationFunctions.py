
import numpy as np
import random



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

    if out_H > H or out_W > W or out_D > D:
        image, mask = pad_to_minimum_shape(image, mask, output_shape)
        H, W, D = image.shape

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
    # Get all tumor voxel coordinates (y, x, z)
    tumor_coords = coords.T.tolist()
    random.shuffle(tumor_coords)

    selected_centers = []
    min_dist = min(out_H, out_W, out_D) * 0.5  # adjust this to control spacing

    def is_far_enough(new_center):
        for cy, cx, cz in selected_centers:
            dist = np.sqrt((cy - new_center[0])**2 + (cx - new_center[1])**2 + (cz - new_center[2])**2)
            if dist < min_dist:
                return False
        return True

    # --- Tumor crops ---
    for cy, cx, cz in tumor_coords:
        if len(crops) >= num_tumor:
            break
        if not is_far_enough((cy, cx, cz)):
            continue
        img_crop, msk_crop = extract_crop(cy, cx, cz)
        if img_crop is not None and np.any(msk_crop):
            crops.append((img_crop, msk_crop))
            selected_centers.append((cy, cx, cz))


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


def pad_to_minimum_shape(image, mask, min_shape):
    """
    Pads image and mask with zeros to ensure they are at least `min_shape` in each dimension.
    Assumes shape is (H, W, D).
    """
    H, W, D = image.shape
    pad_H = max(0, min_shape[0] - H)
    pad_W = max(0, min_shape[1] - W)
    pad_D = max(0, min_shape[2] - D)

    pad_before = (pad_H // 2, pad_W // 2, pad_D // 2)
    pad_after  = (pad_H - pad_before[0], pad_W - pad_before[1], pad_D - pad_before[2])

    pad_width = (
        (pad_before[0], pad_after[0]),
        (pad_before[1], pad_after[1]),
        (pad_before[2], pad_after[2])
    )

    image_padded = np.pad(image, pad_width, mode='constant', constant_values=0)
    mask_padded = np.pad(mask, pad_width, mode='constant', constant_values=0)
    return image_padded, mask_padded