import torch
from torch import nn

import torchvision.models as tvm

# backbones with reducing linear layer at top

class ResNet50(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'ResNet50_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.resnet50(weights=weights)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nfc = net.fc.in_features
        if out_channels is None:
            net.fc = nn.Identity()
        else:
            net.fc = nn.Linear(nfc, out_channels, bias=True)
        
        self.net = net
        self.nfc = nfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
