import torch
import torch.nn as nn
import torchvision
from typing import List



class ResNetEncoder(nn.Module):
    """
    Feature encoder based on the torchvision models of the ResNet family. The
    models are initialised with weights pretrained for classification on ImageNet-1K. 
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
            backbone_net: Which ResNet architecture to use as backbone; available:
                "resnet18", "resnet34", "resnet50", "resnet50xt", "resnet101",
                "resnet101xt", "resnet152".
            pretrained: If True, models are initialised with the latest version
                of pretrained ImageNet-1K weights. 
            extract_blocks: A list indicating which blocks (1-4) to extract the
                features from. These features are concatenated to form the output.
            out_ft_size: The size of the output features (hidden dimension) after
                1x1-conv projection.
            dropout: Dropout probability in the last layer of the encoder.
        """
        super().__init__()

        if extract_blocks is None:   # Only get output from last backbone's layer
            extract_blocks = [4]

        for block in extract_blocks:
            assert block in [1,2,3,4]
            if (block == 1) and (len(extract_blocks) > 1):
                raise Exception(
                    "Output features from block 1 cannot be concatenated with "
                    "features from other blocks due to shape restrictions."
                )

        self.extract_blocks = sorted(extract_blocks)

        
        # Select latest version of pre-trained weights
        weight_init = "DEFAULT" if pretrained else None


        # Select the type of architecture for the encoder backbone
        if backbone_net == "resnet18":
            base_size = 64
            pretrained_net = torchvision.models.resnet18(
                weights = weight_init,
                progress = True,
            )
        elif backbone_net == "resnet34":
            base_size = 64
            pretrained_net = torchvision.models.resnet34(
                weights = weight_init,
                progress = True,
            )
        elif backbone_net == "resnet50":
            base_size = 256
            pretrained_net = torchvision.models.resnet50(
                weights = weight_init,
                progress = True,
                replace_stride_with_dilation=[False, True, True]
            )
        elif backbone_net == "resnet50xt":
            base_size = 256
            pretrained_net = torchvision.models.resnext50_32x4d(
                weights = weight_init,
                progress = True,
                replace_stride_with_dilation=[False, True, True]
            )
        elif backbone_net == "resnet101":
            base_size = 256
            pretrained_net = torchvision.models.resnet101(
                weights = weight_init,
                progress = True,
                replace_stride_with_dilation=[False, True, True]
            )
        elif backbone_net == "resnet101xt":
            base_size = 256
            pretrained_net = torchvision.models.resnext101_32x8d(
                weights = weight_init,
                progress = True,
                replace_stride_with_dilation=[False, True, True]
            )
        elif backbone_net == "resnet152":
            base_size = 256
            pretrained_net = torchvision.models.resnet152(
                weights = weight_init,
                progress = True,
                replace_stride_with_dilation=[False, True, True]
            )
        else:
            raise ValueError("Invalid value for keyword 'backbone_net'.")


        # Construct the backbone
        self.layer0 = nn.Sequential( 
            pretrained_net.conv1,
            pretrained_net.bn1,
            pretrained_net.relu,
            pretrained_net.maxpool
        )
        self.encoding_layers = nn.ModuleList(
            [pretrained_net.layer1,
            pretrained_net.layer2,
            pretrained_net.layer3,
            pretrained_net.layer4] [:self.extract_blocks[-1]]
        )   # Consider only up to the last indicated block  


        # Hidden dimension depends on concatenation of extracted features from respective block
        # For resnet50: Out_size = 256 (block1), 512 (block2), 1024 (block3), 2048 (block4)
        hid_size = sum([base_size * 2**(n-1) for n in self.extract_blocks])

        # Projection layer to reduce output feature dimension
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
