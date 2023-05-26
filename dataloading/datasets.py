import os
import json
import h5py
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Callable, Any

from dataloading.data_utils import DATASET_INFO, get_normalisation_fn
from dataloading.transforms import get_default_augment_transforms




class Covid19_Dataset(Dataset):
    """
    Main data module for loading and processing operations on the COVID-19-CT-Seg
    dataset. Expects a data folder of patient files in hdf5 format, where each file
    contains the volumetric CT scan of the patient lungs, as well as the segmentation
    masks for the infection, lungs and the union of infection and lungs.
    """
    def __init__(self,      
        data_path: Path,
        data_info_path: Path,
        fold: int,
        mode: str,
        preload: bool=False,
        seg_masks_union: bool=False,
        norm_level: int=3,
        norm_type: int=2,
        repeat_ch: bool=True,
        custom_transforms: Callable=None,
    ):
        """
        Args:
            data_path: Path to dataset directory.
            data_info_path: Path to directory containing json info for each patient.
                Used for image normalisation and local/global index managing.
            fold: Fold number (0-4) to get dataset split for 5-fold cross validation.
                Each split contains 5 patient files used for validation and the
                remaining 15 patient files used for training.
            mode: One of ['train', 'valid', 'test']. Indicates whether data should
                be processed for training or inference.
            preload: If True, the whole dataset is loaded onto RAM at initialisation.
                If False, patient files are loaded individually on the fly.
                Recommended True during training.
            seg_masks_union: If True, the only segmentation mask used is the union
                of lungs and infection. If False, use lungs mask and infection
                mask separately. Default False.
            norm_level: Either 1, 2 or 3, corresponding to slice, patient and
                dataset-level normalisation, respectively. The statistics for
                patient and datataset-level normalisation are gathered from the
                patient's json file. Default 3.
            norm_type: Either 1 or 2, corresponding to min-max and mean-stdev
                normalisation, respectively. Default 2.
            repeat_ch: If True, stack a copy of the data item three times along the
                channel dimension. Used for pre-trained PyTorch backbones that only
                accept input tensors with a channel size of 3.
            custom_transforms: Optional composition of torchvision transformations 
                to be applied for augmentation during data serving (training only).
                The composition is expected to take in and transform both the CT
                image and corresponding masks simultaneously. If none is provided,
                default augmentation transformations will be applied.
        """
        super().__init__()
        
        assert mode in ["train", "valid", "test"], "Invalid mode."

        self.data_path = data_path
        self.data_info_path = data_info_path
        self.mode = mode
        self.preload = preload
        self.repeat_ch = repeat_ch

        self.patient_list = sorted(os.listdir(data_path))
        self.patient_ids = [
            x.split('.')[0] for x in self.patient_list if x.endswith(".h5")
        ]
        
        # Get training/validation split for given fold
        fold_idxs = DATASET_INFO["folds"][fold]["fold_idxs"]
        if mode == "train":
            self.patient_ids = [
                elem for idx, elem in enumerate(self.patient_ids) if idx not in fold_idxs
            ]
        else:
            self.patient_ids = [
                elem for idx, elem in enumerate(self.patient_ids) if idx in fold_idxs
            ]

        # Select which type of masks to use (union of infection and lungs or separate)
        self.seg_masks = [
            DATASET_INFO["seg_masks"][2]
        ] if seg_masks_union else DATASET_INFO["seg_masks"][:2]


        self.normalise = get_normalisation_fn(norm_level, norm_type)

        if custom_transforms:
            self.transforms = custom_transforms
        else:
            self.transforms = get_default_augment_transforms()


        # Compile indexing look-up framework
        self.dataset_idx_info = self.dataset_indexing()

        if self.preload:
            self.dataset = self.dataset_preload()


    
    def __len__(self) -> int:
        return len(self.dataset_idx_info["local_idx"])



    def dataset_indexing(self) -> Dict[str, List[Any]]:
        """
        Compiles an indexing framework to easily access info about any 2D data
        slice within the current dataset split.
        """

        dataset_idx_info = dict(
            map(lambda x: (x, []),
                ["patient_id", "local_idx", "n_slices", "labels"])
        )
        
        for pid in tqdm(self.patient_ids, desc="Compiling indexing framework"):

            # Load total number of slices in current patient's scan
            with open(os.path.join(self.data_info_path, f"{pid}.json"), 'r') as f_info:
                img_info = json.load(f_info)
                n_slices = img_info["img_shape"][0]
                f_info.close()
                
            # Keep track of local slice index within current volume
            for zslice in range(n_slices):       
                dataset_idx_info["patient_id"].append(pid)
                dataset_idx_info["local_idx"].append(zslice)
                dataset_idx_info["n_slices"].append(n_slices)

                # Construct a one-hot-like vector with length equal to the
                # number of segmentation labels to indicate the presence of
                # an annotated label in the current slice.
                # Assumes labels are already mapped to [1, 2, ..., n_labels]
                lbls_onehot = np.zeros(len(DATASET_INFO["seg_labels"].keys()))
                for lb_id, lb in DATASET_INFO["seg_labels"].items():
                    if zslice in img_info["annotated_slices"][lb_id]:
                        lbls_onehot[lb-1] = 1

                dataset_idx_info["labels"].append(lbls_onehot)

        return dataset_idx_info
                


    def data_loader(
        self, patient_id: str, slice_range: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load onto RAM the CT scan and segmentation masks of the patient with
        "patient_id" identifier. The portion of volume loaded is determined by
        the parameter slice_range, so this can be used for both pre-loading and
        on-the-fly per-slice loading. The loaded CT scan is normalised.
        """

        z_start, z_end = slice_range  # Only data slices in this range are loaded

        with h5py.File(os.path.join(self.data_path, f"{patient_id}.h5"), 'r') as f_data:
            # 3D CT scan, shape (Z, H, W)
            img = np.array(f_data["Data"][z_start:z_end]).astype('float32')

            # 3D Segmentation masks, each of shape (Z, H, W)
            masks = []
            for m in self.seg_masks:
                masks.append(np.array(f_data[m][z_start:z_end]).astype('int32'))

            # Stack all masks along the channel dimension C into a single
            # 4D array of shape (Z, C, H, W) 
            masks = np.stack(masks, axis=1)

            # json info for patient data
            with open(os.path.join(self.data_info_path, f"{patient_id}.json"), 'r') as f_info:
                img_info = json.load(f_info)
                f_info.close()

            img = self.normalise(img, img_info).astype('float32')

        return img, masks



    def dataset_preload(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Loads the entire dataset split onto RAM.

        The output data is formatted as a dictionary where each item is a
        (img, masks) pair corresponding to the patient's volume data.
        """

        data_out = {}
        global_slice_idx = 0

        # Load normalised CT scan volume and segmentation masks for each patient
        for pid in tqdm(self.patient_ids, desc="Pre-loading dataset"):
            
            # Get total number of slices in the volume
            n_slices = self.dataset_idx_info["n_slices"][global_slice_idx]
            slice_range = (0, n_slices)  

            img, masks = self.data_loader(pid, slice_range)
            data_out[pid] = (img, masks)

            global_slice_idx += n_slices  # Start of next volume
        
        return data_out



    def binarize_and_select_mask(
        self, masks: np.ndarray, label: int
    ) -> np.ndarray:
        """
        Binarize the masks array according to given label value and return the
        mask along the mask-channel dimension that contains that label.

        Assumes each mask contains disjoint sets of labels, e.g.
        - Lung_Mask contains labels 1 and 2;
        - Infection_Mask contains label 3.

        Args:
            masks: Segmentation masks array, shape (1, C, H, W), C=number of masks
            label: Annotation label, a int in [1, 2, ..., n_labels]

        Return:
            Selected mask array, shape (1, H, W)
        """

        binarized_masks = (masks == int(label)).astype('int32')
        
        if binarized_masks.sum() == 0:
            # Slice doesn't contain label, return any mask
            nonzero_mask_idx = 0
            
        else:   
            # If labels sets are disjoint, all masks except one are zero everywhere
            nonzero_mask_idx = np.nonzero(binarized_masks)[1][0]

        return binarized_masks[:, nonzero_mask_idx]
    


    def __getitem__(self, item: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a CT image slice and corresponding segmentation mask based on
        the index-label pair sampled by the custom few-shot sampler.      
        """

        idx, label = item    # idx is global

        # From indexing framework retrieve patient id and slice index
        # within patient's volume corresponding to "idx"
        patient_id = self.dataset_idx_info["patient_id"][idx]
        local_idx = self.dataset_idx_info["local_idx"][idx]

        # Get slice from preloaded dataset or load on the fly
        if self.preload:
            img = self.dataset[patient_id][0][local_idx : local_idx+1]
            masks = self.dataset[patient_id][1][local_idx : local_idx+1]
        else:
            img, masks = self.data_loader(patient_id, (local_idx, local_idx+1))

        # Get binary mask containing sampled label
        mask = self.binarize_and_select_mask(masks, label)

        img = torch.from_numpy(img)   # Shape (1, H, W)
        mask = torch.from_numpy(mask)    # Shape (1, H, W)
        
        if self.repeat_ch:   # Shape (3, H, W)
            img = img.repeat_interleave(3, dim=0)
        
        if self.mode == "train":
            img, mask  = self.transforms((img, mask))

        # Add batch dimension
        img, mask = img.unsqueeze(0), mask.unsqueeze(0)

        return img, mask


