from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from typing import Type, Union, List, Optional, Callable
from torch import Tensor
import torch
import torch.nn as nn

from functools import partial


class BiasLayer(torch.nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        bias_value = torch.zeros((dim))
        self.bias_layer = torch.nn.Parameter(bias_value)
    
    def forward(self, x):
        return x + self.bias_layer
    

class CustomResNet(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
          block,
          layers,
          num_classes,
          zero_init_residual,
          groups,
          width_per_group,
          replace_stride_with_dilation,
          norm_layer
      )
        

    def forward2(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x
    
class ResNetCustomBias(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
          block,
          layers,
          num_classes,
          zero_init_residual,
          groups,
          width_per_group,
          replace_stride_with_dilation,
          norm_layer
      )
        
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=False)
        self.bias_layer = BiasLayer(num_classes)

    
    def forward(self, x: Tensor) -> Tensor:
        x = self.forward2(x)
        x = self.bias_layer(x)
        return x
        

    def forward2(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=N_GROUPS, num_channels=num_channels,
                                 eps=1e-5, affine=True)
    
    def forward(self, x):
        x = self.norm(x)
        return x
    
class MyLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyLayerNorm, self).__init__()
        # LayerNorm normalizes across the channel dimension, so we set normalized_shape to the number of channels.
        self.norm = nn.LayerNorm(normalized_shape=num_channels, eps=1e-5, elementwise_affine=True)
    
    def forward(self, x):
        # Flatten spatial dimensions so LayerNorm can apply across the channels
        # Shape: [batch_size, num_channels, height, width] -> [batch_size, num_channels]
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

CustomResNetPartial = partial(CustomResNet, block=BasicBlock, layers=[2, 2, 2, 2])
ResNetCustomBiasPartial = partial(ResNetCustomBias, block=BasicBlock, layers=[2, 2, 2, 2])