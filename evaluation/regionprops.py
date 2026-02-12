import cv2
from pollen_datasets.dataset_builder.holographic_features import regionprops_from_numpy

def calculate_regionprops(df, i, full_path, resolution=(200, 200)):

    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

    # Resize back to original resolution
    img = cv2.resize(
        img,
        resolution,
        interpolation=cv2.INTER_AREA
    )

    rp = regionprops_from_numpy(img)
    df.at[i, 'area']                  = rp[0].area
    df.at[i, 'bbox_area']             = rp[0].bbox_area
    df.at[i, 'convex_area']           = rp[0].convex_area
    df.at[i, 'major_axis_length']     = rp[0].major_axis_length
    df.at[i, 'minor_axis_length']     = rp[0].minor_axis_length
    df.at[i, 'eccentricity']          = rp[0].eccentricity
    df.at[i, 'solidity']              = rp[0].solidity
    df.at[i, 'perimeter']             = rp[0].perimeter
    df.at[i, 'perimeter_crofton']     = rp[0].perimeter_crofton
    df.at[i, 'equivalent_diameter']   = rp[0].equivalent_diameter
    df.at[i, 'orientation']           = rp[0].orientation
    df.at[i, 'feret_diameter_max']    = rp[0].feret_diameter_max
    df.at[i, 'max_intensity']         = rp[0].intensity_max[0]
    df.at[i, 'min_intensity']         = rp[0].intensity_min[0]
    df.at[i, 'mean_intensity']        = rp[0].intensity_mean[0]