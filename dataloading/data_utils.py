import numpy as np
from typing import Dict, Callable, Any


DATASET_INFO = {
    "seg_masks":
        ["Infection_Mask", "Lung_Mask", "Lung_and_Infection_Mask"],

    "seg_labels":
        {"left_lung": 1, "right_lung": 2, "infection": 3},
    
    "folds":
        {
            0: {"fold_idxs": list(range(0, 5)), "support_idx": 0},
            1: {"fold_idxs": list(range(4, 9)), "support_idx": 0},
            2: {"fold_idxs": list(range(8, 13)), "support_idx": 0},
            3: {"fold_idxs": list(range(12, 17)), "support_idx": 0},
            4: {"fold_idxs": [0] + list(range(16, 20)), "support_idx": 1},
        }
}



def get_normalisation_fn(norm_level:int, norm_type:int) -> Callable:
    """
    Selects the normalisation function to be applied to input images.

    Args:
        norm_level: Either 2 or 3, corresponding to patient and dataset-level
            normalisation, respectively. The statistics are gathered from the
            patient's json file. If different than 2 or 3, the function computes
            the normalisation statistics on the fly (used for slice-level
            normalisation).
        norm_type: If 1, selects min-max normalisation, else, selects mean-stdev
            normalisation.

    Returns:
        The corresponding normalisation function (callable object).
    """

    json_stats = ""
    generic = False

    if norm_level == 2:
        json_stats += "patient_"
    elif norm_level == 3:
        json_stats += "global_"
    else:
        generic = True

    if norm_type == 1:
        json_stats += "min_max"
    else:
        json_stats += "mean_std"

    def normalisation_fn(img: np.ndarray, img_info: Dict[str, Any]) -> np.ndarray:
        """
        Args:
            img: Un-normalised CT scan.
            img_info: Patient's json dictionary with normalisation statistics.

        Returns:
            Normalised CT scan.
        """
        # Min-Max normalisation
        if norm_type == 1:
            min_val, max_val = (
                np.min(img), np.max(img)
            ) if generic else img_info[json_stats]

            return  (img - min_val) / (max_val - min_val)

        # Mean-Stdev normalisation
        else:
            mean_val, std_val = (
                np.mean(img), np.std(img)
            ) if generic else img_info[json_stats]

            return  (img - mean_val) / std_val

    return normalisation_fn