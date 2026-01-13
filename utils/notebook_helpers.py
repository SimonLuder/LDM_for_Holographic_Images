import numpy as np
import pandas as pd

def compare_min_max(df, pred_suffix="_pred", gt_suffix="_gt"):
    rows = []

    gt_cols = [c for c in df.columns if c.endswith(gt_suffix)]

    for gt_col in gt_cols:
        base = gt_col[: -len(gt_suffix)]
        pred_col = base + pred_suffix

        if pred_col not in df.columns:
            continue

        gt_min, gt_med, gt_max = df[gt_col].min(), df[gt_col].median(), df[gt_col].max()
        pred_min, pred_med, pred_max = df[pred_col].min(), df[pred_col].median(), df[pred_col].max()

        rows.append({
            "metric": base,
            "gt_min": gt_min,
            "pred_min": pred_min,
            "gt_median": gt_med,
            "pred_median": pred_med,
            "gt_max": gt_max,
            "pred_max": pred_max,
        })

    return pd.DataFrame(rows)