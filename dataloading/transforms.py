import torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as tr_F
from typing import List, Tuple, Callable


AUGMENT_INFO = {
    "aug_prob": 0.5,

    "flip": { # Half of the COVID-19-CT-Seg dataset is originally flipped vertically
        "vflip_p": 0.5, "hflip_p": 0.0,
    },

    "affine": {
        "angle_range": (-5, 5),
        "translate_range": [(-5, 5), (-5, 5)],
        "scale_range": (0.9, 1.1),
        "shear_range": [(-5, 5), (-5, 5)],
    },

    "elastic": {
        "alpha_range": (15.0, 40.0),
        "sigma_range": (5.0, 5.0),
    },

    "gamma": {
        "gamma_range": (0.5, 1.5),
    }
}



def get_default_augment_transforms() -> Callable:
    """
    Generate a composition of torchvision transformations to augment data on the
    fly. Each transformation is applied according to a given probability.
    """

    # Probability that each transform is applied
    p = AUGMENT_INFO["aug_prob"]

    # Geometric transforms
    flip_transform = RandomFlip(**AUGMENT_INFO["flip"])
    affine_transform = RandomAffine(**AUGMENT_INFO["affine"])
    elastic_transform = RandomElastic(**AUGMENT_INFO["elastic"])

    # Intensity transforms
    gamma_transform = RandomGamma(**AUGMENT_INFO["gamma"])

    # Get transforms based on probability p
    transforms_list = [flip_transform]
    for tr in [affine_transform, elastic_transform, gamma_transform]:
        if np.random.random() < p:
             transforms_list += [tr]

    return transforms.Compose(transforms_list)



class RandomFlip(object):
    """
    Randomly flip input tensors horizontally and/or vertically along the last
    two dimensions based on input probabilities.
    """

    def __init__(self, vflip_p: float, hflip_p: float):
        """
        Args:
            vflip_p: The probability that the input is flipped vertically
            hflip_p: The probability that the input is flipped horizontally
        """

        self.vflip_p = vflip_p
        self.hflip_p = hflip_p

    def __call__(
        self, sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sample: (img, masks) pair, where both img and masks are torch
                tensors with expected shape [..., H, W], where ... means an
                arbitrary number of leading dimensions.
        """
    
        img, masks = sample

        if np.random.random() < self.vflip_p:
            img = tr_F.vflip(img)
            masks = tr_F.vflip(masks)

        if np.random.random() < self.hflip_p:
            img = tr_F.hflip(img)
            masks = tr_F.hflip(masks)

        return img, masks



class RandomAffine(object):
    """
    Apply a random affine transformation. The transformation parameters are
    generated randomly from given input ranges. 
    
    More details on the effect of each parameter on
    https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.affine.html
    """

    def __init__(self,
        angle_range: Tuple[float, float],
        translate_range: List[Tuple[float, float]],
        scale_range: Tuple[float, float],
        shear_range: List[Tuple[float, float]],
    ):

        self.angle_range = angle_range
        self.translate_range = translate_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.interpolation = transforms.InterpolationMode.BILINEAR


    def __call__(
        self, sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sample: (img, masks) pair, where both img and masks are torch
                tensors with expected shape [..., H, W], where ... means an
                arbitrary number of leading dimensions.
        """
    
        img, masks = sample

        # Generate transform parameters
        angle = np.random.uniform(*self.angle_range)
        translate = [
            np.random.uniform(*self.translate_range[0]),
            np.random.uniform(*self.translate_range[1])
        ]
        scale = np.random.uniform(*self.scale_range)
        shear = [
            np.random.uniform(*self.shear_range[0]),
            np.random.uniform(*self.shear_range[1])
        ]
        
        # Apply transform
        fill_val = float(img.min())
        img =  tr_F.affine(img,
            angle, translate, scale, shear, self.interpolation, fill=fill_val
        )
        masks =  tr_F.affine(masks,
            angle, translate, scale, shear, self.interpolation, fill=0
        )

        return img, masks



class RandomElastic(object):
    """
    Apply a random elastic transformation. The transformation parameters are
    generated randomly from given input ranges. 
    
    More details on the effect of each parameter on
    https://pytorch.org/vision/stable/generated/torchvision.transforms.ElasticTransform.html
    """

    def __init__(self,
        alpha_range: Tuple[float, float],
        sigma_range: Tuple[float, float],
    ):

        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.interpolation = transforms.InterpolationMode.BILINEAR


    def __call__(
        self, sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sample: (img, masks) pair, where both img and masks are torch
                tensors with expected shape [..., H, W], where ... means an
                arbitrary number of leading dimensions.
        """
    
        img, masks = sample
        
        # Generate transform parameters
        alpha = np.random.uniform(*self.alpha_range)
        sigma = np.random.uniform(*self.sigma_range)
        elastic_tr = transforms.ElasticTransform(alpha, sigma, self.interpolation)

        # Apply transform
        img =  elastic_tr(img)
        masks = elastic_tr(masks)

        return img, masks      



class RandomGamma(object):
    """
    Apply a random gamma transformation. The transformation parameters are
    generated randomly from given input ranges. 
    
    More details on the effect of each parameter on
    https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.adjust_gamma.html
    """

    def __init__(self, gamma_range: Tuple[float, float]):

        self.gamma_range = gamma_range


    def __call__(
        self, sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sample: (img, masks) pair, where both img and masks are torch
                tensors with expected shape [..., H, W], where ... means an
                arbitrary number of leading dimensions.
        """
    
        img, masks = sample

        # Generate transform parameters
        gamma = np.random.uniform(*self.gamma_range)

        # Apply transform (only to img since this is intensity transform)
        img_min, img_max = img.min(), img.max() 
        intensity_range = img_max - img_min + 1e-5
        img -= img_min - 1e-5   # Pixel values must be non-negative
        img = intensity_range * torch.pow(img/intensity_range,  gamma)
        img += img_min    # Shift intensity to original range

        return img, masks
