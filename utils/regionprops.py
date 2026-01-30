import numpy as np

from pollen_datasets.dataset_builder.holographic_features import regionprops_from_numpy

def regionprops_from_torch(img_torch):
    """
    img_torch: torch.Tensor (H, W) or (1, H, W)
    returns: regionprops list
    """

    if img_torch.ndim == 3:
        img_torch = img_torch.squeeze(0)

    assert img_torch.ndim == 2, "Expected (H, W) or (1, H, W)"

    img_np = img_torch.detach().cpu().numpy()

    # ensure uint8 if needed by OpenCV
    if img_np.dtype != np.uint8:
        img_np = (255 * img_np).clip(0, 255).astype(np.uint8)

    return regionprops_from_numpy(img_np)


def regionprops_from_torch_batch(img_batch):
    """
    img_batch: torch.Tensor (N, 1, H, W)
    returns: list of regionprops (length N)
    """

    assert img_batch.ndim == 4 and img_batch.shape[1] == 1

    img_batch = img_batch.detach().cpu()

    batch_rp = []

    for img in img_batch:
        # img: (1, H, W)
        rp = regionprops_from_torch(img)
        batch_rp.append(rp)

    return batch_rp


def regionprops_batch_mean_dict(rps_batch):
    """
    rps_batch: list of regionprops lists (length N)

    returns: dict with mean values across batch
    """

    # collect per-image values
    collected = {
        "area": [],
        "bbox_area": [],
        "convex_area": [],
        "major_axis_length": [],
        "minor_axis_length": [],
        "eccentricity": [],
        "solidity": [],
        "perimeter": [],
        "perimeter_crofton": [],
        "equivalent_diameter": [],
        "orientation": [],
        "feret_diameter_max": [],
        "max_intensity": [],
        "min_intensity": [],
        "mean_intensity": [],
    }

    for rps in rps_batch:
        if len(rps) == 0:
            continue

        rp = rps[0]

        collected["area"].append(rp.area)
        collected["bbox_area"].append(rp.bbox_area)
        collected["convex_area"].append(rp.convex_area)
        collected["major_axis_length"].append(rp.major_axis_length)
        collected["minor_axis_length"].append(rp.minor_axis_length)
        collected["eccentricity"].append(rp.eccentricity)
        collected["solidity"].append(rp.solidity)
        collected["perimeter"].append(rp.perimeter)
        collected["perimeter_crofton"].append(rp.perimeter_crofton)
        collected["equivalent_diameter"].append(rp.equivalent_diameter)
        collected["orientation"].append(rp.orientation)
        collected["feret_diameter_max"].append(rp.feret_diameter_max)
        collected["max_intensity"].append(float(rp.intensity_max[0]))
        collected["min_intensity"].append(float(rp.intensity_min[0]))
        collected["mean_intensity"].append(float(rp.intensity_mean[0]))

    mean_dict = {
        k: float(np.nanmean(v)) if len(v) > 0 else np.nan
        for k, v in collected.items()
    }

    return mean_dict


def angular_diff_rad(a, b, period=np.pi):
    """
    Minimal absolute angular difference (radians)
    Handles wrap-around correctly.

    a, b: angles in radians
    period: np.pi for regionprops.orientation
    """
    return abs((a - b + period / 2) % period - period / 2)


def abs_prop_rps_error(batch_gt, batch_gen, eps=1e-8, reduction='mean'):
    """
    batch_gt, batch_gen:
        list of regionprops lists (length N)
    reduction: 'mean' or None

    returns:
        dict: mean normalized error per property over batch
    """

    assert len(batch_gt) == len(batch_gen)

    errors = {
        "area": [],
        "bbox_area": [],
        "convex_area": [],
        "major_axis_length": [],
        "minor_axis_length": [],
        "eccentricity": [],
        "solidity": [],
        "perimeter": [],
        "perimeter_crofton": [],
        "equivalent_diameter": [],
        "orientation": [],
        "feret_diameter_max": [],
        "max_intensity": [],
        "min_intensity": [],
        "mean_intensity": [],
    }

    for rps_gt, rps_gen in zip(batch_gt, batch_gen):
        if len(rps_gt) == 0 or len(rps_gen) == 0:
            continue

        gt = rps_gt[0]
        gen = rps_gen[0]

        errors["area"].append(abs(gen.area - gt.area) / (abs(gt.area) + eps))
        errors["bbox_area"].append(abs(gen.bbox_area - gt.bbox_area) / (abs(gt.bbox_area) + eps))
        errors["convex_area"].append(abs(gen.convex_area - gt.convex_area) / (abs(gt.convex_area) + eps))
        errors["major_axis_length"].append(abs(gen.major_axis_length - gt.major_axis_length) / (abs(gt.major_axis_length) + eps))
        errors["minor_axis_length"].append(abs(gen.minor_axis_length - gt.minor_axis_length) / (abs(gt.minor_axis_length) + eps))
        errors["eccentricity"].append(abs(gen.eccentricity - gt.eccentricity))
        errors["solidity"].append(abs(gen.solidity - gt.solidity))
        errors["perimeter"].append(abs(gen.perimeter - gt.perimeter) / (abs(gt.perimeter) + eps))
        errors["perimeter_crofton"].append(abs(gen.perimeter_crofton - gt.perimeter_crofton) / (abs(gt.perimeter_crofton) + eps))
        errors["equivalent_diameter"].append(abs(gen.equivalent_diameter - gt.equivalent_diameter) / (abs(gt.equivalent_diameter) + eps))
        errors["orientation"].append(angular_diff_rad(gen.orientation, gt.orientation))
        errors["feret_diameter_max"].append(abs(gen.feret_diameter_max - gt.feret_diameter_max) / (abs(gt.feret_diameter_max) + eps))
        errors["max_intensity"].append(abs(gen.intensity_max[0] - gt.intensity_max[0]) / (abs(gt.intensity_max[0]) + eps))
        errors["min_intensity"].append(abs(gen.intensity_min[0] - gt.intensity_min[0]) / (abs(gt.intensity_min[0]) + eps))
        errors["mean_intensity"].append(abs(gen.intensity_mean[0] - gt.intensity_mean[0]) / (abs(gt.intensity_mean[0]) + eps))

    # Aggregate across batch
    if reduction == 'mean':
        mean_error_dict = {
            k: float(np.nanmean(v)) if len(v) > 0 else np.nan
            for k, v in errors.items()
        }
        return mean_error_dict
    
    return errors