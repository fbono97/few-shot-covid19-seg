import torch
import torch.nn as nn
from typing import Dict, Any

from models.panet import PANet
from models.adnet import ADNet
from models.alpnet import ALPNet
from models.backbones.resnet_backbone import ResNetEncoder
from models.backbones.efficientnet_backbone import EfficientNetEncoder



def get_model_by_name(
    model_name: str, device: torch.device, mode: str, cfg: Dict[str, Any],
) -> nn.Module:
    """
    Returns chosen model initialised with configuration parameters.
    
    Args:
        model_name: Either "PANet", "ADNet" or "ALPNet".
        device: The torch device where to send the model.
        mode: One of ['train', 'valid', 'test'].
        cfg: Configuration dictionary containing model parameters.
    """

    if mode in ["valid", "test"]:
        cfg["pretrained"] = False
        cfg["n_way"] = 1
        cfg["prototype_alignment"] = False
        

    # Retrieve feature encoder
    if "resnet" in cfg["backbone_net"]:
        ft_encoder = ResNetEncoder(backbone_net = cfg["backbone_net"],
                                   pretrained = cfg["pretrained"],
                                   extract_blocks = cfg["extract_blocks"],
                                   out_ft_size = cfg["out_ft_size"],
                                   dropout = cfg["dropout"])
        
    elif "efficientnet" in cfg["backbone_net"]:
        ft_encoder = EfficientNetEncoder(backbone_net = cfg["backbone_net"],
                                         pretrained = cfg["pretrained"],
                                         extract_blocks = cfg["extract_blocks"],
                                         out_ft_size = cfg["out_ft_size"],
                                         dropout = cfg["dropout"])
    else:
        raise ValueError("Feature encoder must be from the ResNet or EfficientNet families.")


    # Initialise the few-shot model
    if model_name == "PANet":
        model = PANet(n_way = cfg["n_way"],
                      k_shot = cfg["k_shot"],
                      encoder = ft_encoder.to(device),
                      prototype_alignment = cfg["prototype_alignment"])
        
    elif model_name == "ADNet":
        model = ADNet(n_way = cfg["n_way"],
                      k_shot = cfg["k_shot"],
                      encoder = ft_encoder.to(device),
                      prototype_alignment = cfg["prototype_alignment"])
        
    elif model_name == "ALPNet":
        model = ALPNet(n_way = cfg["n_way"],
                       k_shot = cfg["k_shot"],
                       encoder = ft_encoder.to(device),
                       pool_window = cfg["pool_window"],
                       fg_threshold = cfg["fg_threshold"],
                       bg_threshold = cfg["bg_threshold"],
                       prototype_alignment = cfg["prototype_alignment"])     
        
    else:
        raise ValueError("Invalid model name.")
 
    return model
    
    


class ModelLoss(nn.Module):
    """
    Custom loss module to account for output differences in models PANet,
    ADNet and ALPNet.
    """
    def __init__(self, model_name: str, weights: torch.Tensor = None):
        super(ModelLoss, self).__init__()

        self.model_name = model_name

        if model_name in ["PANet", "ALPNet"]:
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        elif model_name == "ADNet":
            self.criterion = nn.NLLLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor):

        if self.model_name == "ADNet":
            eps = torch.finfo(torch.float32).eps
            output = torch.log(torch.clamp(output, eps, 1-eps))
            
        return self.criterion(output, target)