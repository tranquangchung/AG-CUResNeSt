""" Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`
Attributes:
    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)
Methods:
    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).
        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)
        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""

import torch.nn as nn
import resnest
from pretrainedmodels.models.torchvision_models import pretrained_settings
import pdb
from ._base import EncoderMixin
from resnest.torch import resnest50
from resnest.torch import resnest101
from models.base.modules import SCSEModule, ASPP 
from models.base.initialization import initialize_module, initialize_head

class ResNestEncoder50(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnest50(pretrained=True)
        self._out_channels = 1
        self._in_channels = 3
        self._depth = 5
        self.out_channels = (3, 64, 256, 512, 1024, 2048)
        del self.encoder.fc
        del self.encoder.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu),
            nn.Sequential(self.encoder.maxpool, self.encoder.layer1),
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")
        super().load_state_dict(state_dict, **kwargs)
    
    def load_state_dict_2(self, state_dict, **kwargs):
        super().load_state_dict(state_dict, **kwargs)


class ResNestEncoder101(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnest101(pretrained=True)
        self._out_channels = 1
        self._in_channels = 3
        self._depth = 5
        self.out_channels = (3, 128, 256, 512, 1024, 2048)
        del self.encoder.fc
        del self.encoder.avgpool


    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu),
            nn.Sequential(self.encoder.maxpool, self.encoder.layer1),
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features


    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")
        super().load_state_dict(state_dict, **kwargs)
    
    def load_state_dict_2(self, state_dict, **kwargs):
        super().load_state_dict(state_dict, **kwargs)

class ResNestEncoder200(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnest101(pretrained=True)
        self._out_channels = 1
        self._in_channels = 3
        self._depth = 5
        self.out_channels = (3, 128, 256, 512, 1024, 2048)
        del self.encoder.fc
        del self.encoder.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu),
            nn.Sequential(self.encoder.maxpool, self.encoder.layer1),
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        return features


    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")
        super().load_state_dict(state_dict, **kwargs)
    
    def load_state_dict_2(self, state_dict, **kwargs):
        super().load_state_dict(state_dict, **kwargs)
