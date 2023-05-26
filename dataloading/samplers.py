import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from typing import List, Tuple, Iterator

from dataloading.data_utils import DATASET_INFO



class FewShotSampler(Sampler):
    """
    Custom sampler for few-shot settings. At each iteration, it samples dataset
    indices for batches of n-way-k-shot episodic tasks.
    """
    def __init__(self,
        dataset: Dataset,
        n_way: int,
        k_shot: int,
        n_query: int,
        batch_size: int,
        n_tasks: int,
    ):
        """
        Args:
            dataset: Dataset class used in conjunction with the sampler. The
                class attribute "dataset_idx_info" is used to retrieve the
                dataset indices corresponding to a sampled label.
            n_way: Number of classes (labels) to sample in one task.
            k_shot: Number of support images to sample for each class in one task.
            n_query: Number of query images to sample for each class in one task.
            batch_size: Number of n-way-k-shot tasks to sample in one iteration.
            n_tasks: Total number of iterations (episodes).
        """
        super().__init__(data_source=None)
        
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.batch_size = batch_size
        self.n_tasks = n_tasks
        
        # One-hot-like array to check which slice has which label,
        # shape (len(dataset), number of labels)
        self.lbls_onehot = np.stack(dataset.dataset_idx_info["labels"])
        self.labels = list(DATASET_INFO["seg_labels"].values())


    def __len__(self) -> int:
        return self.n_tasks


    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:

        for _ in range(self.n_tasks):
            
            iter_items = []
            for _ in range(self.batch_size):  # Sample an (n_way-k_shot) task
                
                # In each task, sample n_way segmentation labels
                sampled_lbls = list(np.random.choice(
                    self.labels, size=min(self.n_way, len(self.labels)), replace=False
                ))

                # For each sampled label, sample k_shot + n_query dataset indices
                # containing that label
                for lb in sampled_lbls:
    
                    idxs_list = np.nonzero(self.lbls_onehot[:, lb-1])[0]
                    sampled_idxs = list(np.random.choice(
                        idxs_list, size=self.k_shot + self.n_query, replace=False
                    ))
    
                    iter_items += list(
                        zip(sampled_idxs, [lb] * len(sampled_idxs))
                    )

            # At each iter return a list of sampled (idx, label) pairs
            yield iter_items


    def episodic_collate_fn(
        self, data_batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate function to be used as argument for the collate_fn parameter of
        PyTorch dataloaders. Reshapes the input data into support and query 
        image-mask pairs.
        
        Based on https://github.com/sicara/easy-few-shot-learning/blob/master/easyfsl/samplers/task_sampler.py#L78

        Args:
            data_batch: Input data loaded from the dataset with indices supplied
                by the few-shot sampler. A list of tuples where each tuple contains
                an image slice and corresponding binary mask.

                    - Length of list:  batch_size x n_way x (k_shot + n_query)
                    - Image:  torch.Tensor of shape (ch, H, W), ch=1 or 3
                    - Mask:  torch.Tensor of shape (1, H, W)

        Return:
            - support_imgs:  torch.Tensor of shape (batch_size, n_way, k_shot, ch, H, W)
            - support_masks:  torch.Tensor of shape (batch_size, n_way, k_shot, 1, H, W)
            - query_imgs:  torch.Tensor of shape (batch_size, n_way x n_query, ch, H, W)
            - query_masks:  torch.Tensor of shape (batch_size x n_way x n_query, H, W)
        """

        imgs_batch = torch.cat([x[0] for x in data_batch])
        masks_batch = torch.cat([x[1] for x in data_batch])

        img_shape = imgs_batch.shape[-2:]    # (H, W)
        ch = imgs_batch.shape[-3]

        imgs_batch = imgs_batch.reshape(
            self.batch_size, self.n_way, self.k_shot + self.n_query, ch, *img_shape
        )
        masks_batch = masks_batch.reshape(
            self.batch_size, self.n_way, self.k_shot + self.n_query, 1, *img_shape
        )
        
        support_imgs = imgs_batch[:, :, :self.k_shot]
        support_masks = masks_batch[:, :, :self.k_shot]
        
        query_imgs = imgs_batch[:, :, self.k_shot:].reshape(
            self.batch_size, -1, ch, *img_shape
        )
        query_masks = masks_batch[:, :, self.k_shot:].reshape(-1, *img_shape)

        return (support_imgs,
                support_masks,
                query_imgs,
                query_masks)
        
        
        


class InferenceSampler(Sampler):
    """
    Custom sampler for inference in few-shot segmentation.
    
    In each validation fold split, one patient is used to construct the support
    set, and the remaining 4 patient volumes form the query set. Since the infection
    segmentation mask is non-continuos along the z-axis, here inference is based
    on Evaluation Protocol 2 (EP2) in [1]. That is, inference is treated as a
    series of 1-way-k-shot tasks for each class, where in each task k-shot slices
    are extracted from the support volume and used to segment the whole query volume.
    This is done for all volumes in the query set.
    
    [1] Hansen, S. et al. (2022), "Anomaly detection-inspired few-shot medical
        image segmentation through self-supervision with supervoxels".
        In: Medical Image Analysis, Vol 78, p. 102385.
    """
    def __init__(self,
        dataset: Dataset,
        fold: int,
        k_shot: int,
    ):
        """
        Args:
            dataset: Dataset class used in conjunction with the sampler. The
                class attribute "dataset_idx_info" is used to retrieve the
                dataset indices corresponding to the current test label and the
                dataset indices corresponding to each patient id.
            fold: Fold number (0-4) to get the support volume in the current
                validation fold.
            k_shot: Number of slices to extract from the support volume.
        """
        super().__init__(data_source=None)

        self.k_shot = k_shot
        self.n_way = 1
        self.batch_size = 1    # Used to match dimensions within model
        
        # Split fold into support and query sets
        support_idx = DATASET_INFO["folds"][fold]["support_idx"]
        self.support_id = dataset.patient_ids[support_idx] 
        self.query_ids = [pid for pid in dataset.patient_ids if pid != self.support_id]

        # Retrieve global list of per-slice patient id and labels one-hot array 
        self.patient_ids_glob = np.stack(
            dataset.dataset_idx_info["patient_id"]  # shape (len(dataset))
        )
        self.lbls_onehot = np.stack(
            dataset.dataset_idx_info["labels"]  # shape (len(dataset), n# of labels)
        )

        self.labels = list(DATASET_INFO["seg_labels"].values())
        self.current_label = None  # This will get updated during validation/testing


    def __len__(self) -> int:
        return len(self.query_ids)


    def set_current_label(self, label: int):
        assert label in self.labels
        self.current_label = label


    def get_support_items(self) -> List[Tuple[int, int]]:
        """
        Returns a list of k-shot index-label pairs to be used as support for the
        current label. The returned indices correspond to slices that are
        equidistantly spaced in the support volume.
        """

        # Condition arrays for slices 1) from current patient and 2) containing current label
        id_condition = self.patient_ids_glob == self.support_id
        lbl_condition = self.lbls_onehot[:, self.current_label-1] == 1

        # Extract global indices for slices that meet both conditions
        glob_idxs = np.nonzero(np.bitwise_and(id_condition, lbl_condition))[0]

        # Select k-shot slices (equidistantly in index space) to form the support 
        idx_splits = np.linspace(0, len(glob_idxs), self.k_shot + 1)
        mid_points = (idx_splits[1:] + idx_splits[:-1]) / 2
        supp_idxs = list(glob_idxs[(mid_points).astype('int')])

        supp_items = list(zip(supp_idxs, [self.current_label] *self.k_shot))

        return supp_items


    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:

        for qry_id in self.query_ids:

            # Each episode consists of k-shot support slices and a whole query volume 
            iter_items = self.get_support_items()

            qry_idxs = np.nonzero(self.patient_ids_glob == qry_id)[0]
            iter_items += list(zip(qry_idxs, [self.current_label] * len(qry_idxs)))

            yield iter_items


    def inference_collate_fn(
        self, data_batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference-analogous of episodic_collate_fn(). Here, n_query is treated
        dynamically depending on the size of the query volume at the current
        sampler's iteration.

        Return:
            - support_imgs:  torch.Tensor of shape (1, 1, k_shot, ch, H, W)
            - support_masks:  torch.Tensor of shape (1, 1, k_shot, 1, H, W)
            - query_imgs:  torch.Tensor of shape (1, len(qry_idxs), ch, H, W)
            - query_masks:  torch.Tensor of shape (len(qry_idxs), H, W)
        """

        imgs_batch = torch.cat([x[0] for x in data_batch])
        masks_batch = torch.cat([x[1] for x in data_batch])

        img_shape = imgs_batch.shape[-2:]    # (H, W)
        ch = imgs_batch.shape[-3]

        imgs_batch = imgs_batch.reshape(
            self.batch_size, self.n_way, -1, ch, *img_shape
        )
        masks_batch = masks_batch.reshape(
            self.batch_size, self.n_way, -1, 1, *img_shape
        )
        
        support_imgs = imgs_batch[:, :, :self.k_shot]
        support_masks = masks_batch[:, :, :self.k_shot]
        
        query_imgs = imgs_batch[:, :, self.k_shot:].reshape(
            self.batch_size, -1, ch, *img_shape
        )
        query_masks = masks_batch[:, :, self.k_shot:].reshape(-1, *img_shape)

        return (support_imgs,
                support_masks,
                query_imgs,
                query_masks)
        
        