import cv2
import numpy as np
from skimage.measure import label, regionprops

from pollen_datasets.dataset_builder import holographic_features as hf



def center_mask(mask):
    """Center an object in an image using the mask centroid.
    """
    lbl = label(mask)
    props = regionprops(lbl)

    if len(props) == 0:
        return mask

    cy, cx = props[0].centroid
    h, w = mask.shape

    shift_y = int(round(h / 2 - cy))
    shift_x = int(round(w / 2 - cx))

    M = np.float32([[1, 0, shift_x],
                    [0, 1, shift_y]])

    centered = cv2.warpAffine(
        mask.astype(np.uint8),
        M,
        (w, h),
        flags=cv2.INTER_NEAREST
    )

    return centered


def get_binary_mask(img):
    """
    Uses holography mask function and keeps only
    the largest connected pollen object.
    Returns binary mask (0,1).
    """

    # Ensure uint8
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    thresh = hf.get_grain_mask_from_holo(img)

    # Find external contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        return np.zeros_like(img, dtype=np.uint8)

    # Keep largest contour only
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 1, thickness=-1)

    return mask


def center_image_and_mask(img, mask):
    """
    Center an object in both image and mask using the mask centroid.

    Parameters
    ----------
    img : ndarray (H,W) or (H,W,3)
        Grayscale or RGB image.
    mask : ndarray (H,W)
        Binary mask of the object.

    Returns
    -------
    img_centered : ndarray
    mask_centered : ndarray
    """

    lbl = label(mask)
    props = regionprops(lbl)

    if not props:
        return img, mask

    # take largest connected component
    p = max(props, key=lambda x: x.area)

    cy, cx = p.centroid
    h, w = mask.shape

    shift_y = int(round(h / 2 - cy))
    shift_x = int(round(w / 2 - cx))

    M = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])

    # apply transform
    img_centered = cv2.warpAffine(img,M,(w, h),flags=cv2.INTER_LINEAR)
    mask_centered = cv2.warpAffine(mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST)

    return img_centered, mask_centered


def rotate_image(img, angle, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=interp)


def mse_inside_mask(img1, img2, mask):
    """
    Compute mean squared error between two images inside the mask region only.
    Supports grayscale and RGB images.
    """
    if img1.ndim == 3:
        diff = np.mean((img1 - img2) ** 2, axis=-1)
    else:
        diff = (img1 - img2) ** 2

    return diff[mask > 0].mean()


def dice_score(mask1, mask2, eps=1e-8):
    """
    Computes Dice similarity between two binary masks.
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    size_sum = mask1.sum() + mask2.sum()

    return (2.0 * intersection + eps) / (size_sum + eps)


def rotation_invariant_particle_metrics(
    img_gt,
    img_pred,
    angle_step=5,
    combined_mask="intersection",
):
    """
    Rotation-invariant comparison.

    1) Finds rotation that maximizes Dice overlap.
    2) Computes MSE on texture using that optimal rotation.

    Args
    ----
    combined_mask : {"intersection", "union", "gt"}
        Region used for texture MSE computation.

    Returns
    -------
    best_dice : float
    best_mse  : float
    """

    mask_gt = get_binary_mask(img_gt)
    mask_pred = get_binary_mask(img_pred)

    # ---- center both ----
    img_gt, mask_gt = center_image_and_mask(img_gt, mask_gt)
    img_pred, mask_pred = center_image_and_mask(img_pred, mask_pred)

    best_dice = -np.inf
    best_rotated_img = None
    best_rotated_mask = None

    # ---- find best rotation using Dice ----
    for angle in range(0, 360, angle_step):

        rotated_mask = rotate_image(mask_pred, angle, interp=cv2.INTER_NEAREST)

        dice = dice_score(mask_gt, rotated_mask)

        if dice > best_dice:
            best_dice = dice
            best_rotated_mask = rotated_mask
            best_rotated_img = rotate_image(
                img_pred, angle, interp=cv2.INTER_LINEAR
            )

    # ---- build mask for texture comparison ----
    if combined_mask == "intersection":
        mask = (mask_gt > 0) & (best_rotated_mask > 0)
    elif combined_mask == "union":
        mask = (mask_gt > 0) | (best_rotated_mask > 0)
    elif combined_mask == "gt":
        mask = (mask_gt > 0)
    else:
        raise ValueError("combined_mask must be: intersection, union, or gt")

    if mask.sum() == 0:
        return best_dice, np.inf

    # ---- compute texture MSE on best rotation ----
    best_mse = mse_inside_mask(img_gt, best_rotated_img, mask)

    return best_dice, best_mse