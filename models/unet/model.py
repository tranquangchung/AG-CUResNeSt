from typing import Optional, Union, List
from .decoder import UnetDecoder, UnetCoupleDecoder, DownSkipConnectionU2, DownSizeU2
from ..encoders import get_encoder, get_encoder_resnest
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead
import torch
import torch.nn as nn
from collections import OrderedDict
from models.base.initialization import initialize_module, initialize_head

class Unet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_depth (int): number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature tensor will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        decoder_attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function to apply after final convolution;
            One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

    Returns:
        ``torch.nn.Module``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
            attention_skip=False,
        )
        
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.initialize()


class CUnet_Resnest(torch.nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        couple_unet = False,
        path_load_checkpoint = None
    ):
        super().__init__()
        self.couple_unet = couple_unet
        self.path_load_checkpoint = path_load_checkpoint
        self.encoder_1 = get_encoder_resnest()

        if couple_unet:
            self.encoder_2 = get_encoder_resnest()

        self.decoder = UnetCoupleDecoder(
            encoder_channels=self.encoder_1.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
            couple_unet = couple_unet,
            path_load_checkpoint = path_load_checkpoint
        )

        self.segmentation_head_1 = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        if couple_unet:
            self.segmentation_head_2 = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=classes,
                activation=activation,
                kernel_size=3,
            )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None
        
        down_half_channel = {
                    "relu": 128, #[64,128] 
                    "layer1": 256,
                    "layer2": 512,
                    "layer3": 1024,
                    "layer4": 2048, #[2048]
                }
        if self.couple_unet:
            down_channel_input_u2 = 3 + decoder_channels[-1] + down_half_channel['relu']
            self.down_conv_input_encoder_u2 = DownSkipConnectionU2(
                    down_channel_input_u2, 128)
            self.down_conv = nn.ModuleDict()
            for layer_name, filter_ in down_half_channel.items():
                self.down_conv[layer_name] = DownSkipConnectionU2(filter_*2, filter_)
            self.shortcut_features = [None, 'relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.encoder_index = {
                        "relu": 1,
                        "layer1": 2,
                        "layer2": 3,
                        "layer3": 4,
                        "layer4": 5
                    }
            self.downsizeU2 = DownSizeU2(3+decoder_channels[-1], 3+decoder_channels[-1]) 
        if self.couple_unet:
            self.encoder_2.encoder.conv1[0]=torch.nn.Conv2d(3+decoder_channels[-1], 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) # modify the input
            
        if self.path_load_checkpoint == None:
            self.initialize()
    def initialize(self):
        initialize_head(self.segmentation_head_1)
        if self.couple_unet:
            initialize_head(self.segmentation_head_2)

    def freeze_param(self ,freeze=True):
        if freeze:
            for params in self.encoder_1.parameters():
                params.requires_grad = False
        else: # unfreeze
            for params in self.encoder_1.parameters():
                params.requires_grad = True
            for params in self.encoder_2.parameters():
                params.requires_grad = True
    
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features_1 = self.encoder_1(x)
        decoder_output_u1, decoder_1 = self.decoder.forward_U1(features_1)
        masks_1 = self.segmentation_head_1(decoder_output_u1["x_1"])
        masks_2 = None
        if self.couple_unet:
            features_2 = self.forward_encoder_U2(x, features_1, decoder_output_u1)
            decoder_output_u2, decoder_2 = self.decoder.forward_U2(features_1, features_2, decoder_1)
            masks_2 = self.segmentation_head_2(decoder_output_u2["x_2"])

        return {
                "U1": masks_1,
                "U2": masks_2
            }

    def forward_encoder_U2(self, x, features_1, decoder_output):
        features_2 = []
        input_U2 = torch.cat((x, decoder_output['x_1']), 1)
        x = input_U2
        if self.couple_unet:
            features_2.append(x)
            for i, (name, child) in enumerate(self.encoder_2.encoder.named_children()): 
                if i == 0:
                    x_u2 = child(x)
                else:
                    x_u2 = child(x_u2)
                if name in self.shortcut_features: # 'relu', 'layer1', 'layer2', 'layer3', 'layer4'
                    join_feature_U1 = torch.cat([x_u2, features_1[self.encoder_index[name]]], dim=1)
                    x_u2 = self.down_conv[name](join_feature_U1)
                    features_2.append(x_u2)
        features_2 = features_2 if len(features_2) > 0 else None
        return features_2
