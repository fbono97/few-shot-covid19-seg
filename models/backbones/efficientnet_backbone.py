import torch
import torch.nn as nn
import torchvision
from typing import List



class EfficientNetEncoder(nn.Module):
    """
    Feature encoder based on the torchvision models of the EfficientNet family.
    The models are initialised with weights pretrained for classification on ImageNet-1K.

    Note that here the forward pass is only allowed up to the fifth block of the
    model architecture, as generally the input image shape (256, 256) is too small
    for deeper feature extraction.
    """
    def __init__(self,
        backbone_net: str,
        pretrained: bool = True, 
        extract_blocks: List[int] = None,
        out_ft_size: int = 256, 
        dropout: float = 0.0
    ):
        """
        Args:
            backbone_net: Which EfficientNet architecture to use as backbone;
                available: "efficientnet_s", "efficientnet_m", "efficientnet_l".
            pretrained: If True, models are initialised with the latest version
                of pretrained ImageNet-1K weights. 
            extract_blocks: A list indicating which blocks (1-5) to extract the
                features from. These features are concatenated to form the output.
            out_ft_size: The size of the output features (hidden dimension) after
                1x1-conv projection.
            dropout: Dropout probability in the last layer of the encoder.
        """
        super().__init__()

        if extract_blocks is None:   # Only get output from last backbone's layer
            extract_blocks = [5]

        for block in extract_blocks:
            assert block in [1,2,3,4,5]
            if (block in [1,2,3]) and (len(extract_blocks) > 1):
                raise Exception(
                    "Output features from blocks 1, 2 or 3 cannot be concatenated "
                    "with features from other blocks due to shape restrictions."
                )

        self.extract_blocks = sorted(extract_blocks)

        
        # Select latest version of pre-trained weights
        weight_init = "DEFAULT" if pretrained else None


        # Select the type of architecture for the encoder backbone
        if backbone_net == "efficientnet_s":
            pretrained_net = torchvision.models.efficientnet_v2_s(
                weights = weight_init,
                progress = True,
            )
        elif backbone_net == "efficientnet_m":
            pretrained_net = torchvision.models.efficientnet_v2_m(
                weights = weight_init,
                progress = True,
            )
        elif backbone_net == "efficientnet_l":
            pretrained_net = torchvision.models.efficientnet_v2_l(
                weights = weight_init,
                progress = True,
            )
        else:
            raise ValueError("Invalid value for keyword 'backbone_net'.")


        # Construct the backbone
        self.layer0 = pretrained_net.features[0]
        
        self.encoding_layers = nn.ModuleList(
            [pretrained_net.features[1],
            pretrained_net.features[2],
            pretrained_net.features[3],
            pretrained_net.features[4],
            pretrained_net.features[5]] [:self.extract_blocks[-1]]
        )   # Consider only up to the last indicated block  


        # Output channels from each block
        out_chs = {"efficientnet_s": [24, 48, 64, 128, 160],
                   "efficientnet_m": [24, 48, 80, 160, 176],
                   "efficientnet_l": [32, 64, 96, 192, 224]}

        # Hidden dimension depends on concatenation of extracted features from respective block
        hid_size = sum([out_chs[backbone_net][n-1] for n in self.extract_blocks])

        # Projection layer
        self.conv_out = nn.Sequential(
            nn.Conv2d(hid_size, out_ft_size, kernel_size=1, stride=1, bias=False),
            nn.Dropout(p=dropout, inplace=False)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.layer0(x)

        out_fts = []
        for block_n, layer in enumerate(self.encoding_layers):
            x = layer(x)
            if block_n + 1 in self.extract_blocks:
                out_fts.append(x)
        
        out_fts = torch.cat(out_fts, dim=1)
        out_fts = self.conv_out(out_fts)

        return out_fts
