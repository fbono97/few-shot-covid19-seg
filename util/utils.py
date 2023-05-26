import os
import yaml
import random
import logging
import torch
import numpy as np
from typing import Dict, Tuple, Any



def set_global_seed(seed: int) -> None:
    """
    Set the starting random seed for Python, NumPy and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

    
def check_mkdir(dir_path: str) -> None:
    """
    Check if a directory exists at dir_path and if not create it.
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)



def load_config(cfg_path: str) -> Dict[str, Any]:
    """
    Load configuration dictionary from yaml config file.
    """
    cfg = {}
    assert os.path.isfile(cfg_path) and cfg_path.endswith('.yaml')

    with open(cfg_path, 'r') as f:
        cfg_file = yaml.safe_load(f)

    for section in cfg_file:
        for k, v in cfg_file[section].items():
            cfg[k] = v

    return cfg


    
def get_logger(name: str) -> logging.Logger:
    """
    Configure logger for stdout operations to console.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.propagate = 0
        handler = logging.StreamHandler()   # Log to console
        formatter = logging.Formatter(
            "[%(asctime)s %(name)s]   %(message)s", "%d-%m-%Y %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
    


def get_remaining_train_time(max_iter: int, current_iter: int, avg_iter_time: float) -> str:
    """
    Calculate estimated remaining training time based on the average time taken
    to process the current iteration and the number of remaining iterations.
    """
    remain_iter = max_iter - current_iter
    remain_time = remain_iter * avg_iter_time
    t_m, t_s = divmod(remain_time, 60)
    t_h, t_m = divmod(t_m, 60)
    return '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
    

    
def get_model_params(model: torch.nn.Module) -> Tuple[str]:
    """
    Calculate total number of trainable and non-trainable parameters in a
    PyTorch model.
    """
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_train = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = train + non_train
    
    # Formatting for readability
    if total > 1e6:
        scaling = 1e6
        unit = 'M'
    elif (total > 1e3) and (total < 1e6):
        scaling = 1e3
        unit = 'K'
    else:
        scaling = 1
        unit = ''
        
    train = f"{train/scaling:.2f}{unit}"
    non_train = f"{non_train/scaling:.2f}{unit}"
    total = f"{total/scaling:.2f}{unit}"
        
    return train, non_train, total
    
    
        
class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
        
class Metrics(object):
    """
    Evaluation metrics for binary semantic segmentation.
    Implemented metrics: Dice score, Precision and Recall.
    """
    def __init__(self):
        # Store metrics for each query volume
        self.patients_dice = []
        self.patients_prec = []
        self.patients_rec = []

    def get_patient_scores(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[float]:
        """
        Computes Dice, Precision, Recall scores given the binary segmentation mask
        predicted by the model for a whole query volume and the corresponding
        binary ground truth mask.

        Args:
            pred: model prediction, a tensor of shape (..., H, W)
            target: ground truth, a tensor of shape (..., H, W)
        
        where ... stands for any number of dimensions.
        """
        
        assert len(torch.unique(pred)) < 3, \
        "Non-binary segmentation masks are not supported."
        
        
        tp = torch.sum((target == 1) * (pred == 1)).float()
        fp = torch.sum((target == 0) * (pred == 1)).float()
        fn = torch.sum((target == 1) * (pred == 0)).float()

        dice = 2 * tp / (2 * tp + fp + fn + 1e-5)
        prec = tp / (tp + fp + 1e-5)
        rec = tp / (tp + fn + 1e-5)

        self.patients_dice.append(dice.item())
        self.patients_prec.append(prec.item())
        self.patients_rec.append(rec.item())

        return dice.item(), prec.item(), rec.item()      
        
        
