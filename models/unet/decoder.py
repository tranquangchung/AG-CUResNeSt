import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md
from collections import OrderedDict
from models.base.initialization import initialize_module, initialize_head
import numpy as np 

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class DecoderAttention(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        if skip_channels != 0:
            self.attention_block = md.AttentionBlock(in_channels, skip_channels, out_channels)
        
            self.conv1 = md.Conv2dReLU(
                in_channels + in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            )
            self.attention1 = md.Attention(attention_type, in_channels=in_channels + in_channels)
        else:
            self.conv1 = md.Conv2dReLU(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            )
            self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x_attention = self.attention_block(x, skip)
            x = torch.cat([x, x_attention], dim=1)
            x = self.attention1(x)
        else:
            x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CoupleDecoderAttention(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            skip_from_U1=False,
            attention_type=None,
    ):
        super().__init__()
        if skip_from_U1:
            in_channel_conv = in_channels + in_channels + out_channels
        else:
            in_channel_conv = in_channels + in_channels
        if skip_channels != 0:
            self.attention_block = md.AttentionBlock(in_channels, skip_channels, out_channels)
        
            self.conv1 = md.Conv2dReLU(
                in_channel_conv,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            )
        elif skip_from_U1:
            self.conv1 = md.Conv2dReLU(
                in_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            )
        else:
            self.conv1 = md.Conv2dReLU(
                in_channels,

                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None, skip_U1=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x_attention, _ = self.attention_block(x, skip)
            x = torch.cat([x, x_attention], dim=1)
        if skip_U1 is not None:
            x = torch.cat([x, skip_U1], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            attention_skip=True,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder # (2048, 1024, 512, 256, 64)

        # computing blocks input and output channels
        head_channels = encoder_channels[0] # 2048
        in_channels = [head_channels] + list(decoder_channels[:-1]) # [2048, 256, 128, 64, 32]
        skip_channels = list(encoder_channels[1:]) + [0] # [1024, 512, 256, 64, 0]
        out_channels = decoder_channels # (256, 128, 64, 32, 16)

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()


        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        if attention_skip:
            blocks = [
                DecoderAttention(in_ch, skip_ch, out_ch, **kwargs)
                for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
            ]
            self.blocks = nn.ModuleList(blocks)
        else:
            blocks = [
                DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
                for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
            ]
            self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class DownSkipConnectionU2(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias=True):
        super(DownSkipConnectionU2, self).__init__()
        self.down_channel_conv = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=1, padding=1, bias=use_bias)
    
    def forward(self, x):
        return self.down_channel_conv(x)

class DownSizeU2(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias=True):
        super(DownSizeU2, self).__init__()
        self.down_size = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=2, padding=1, bias=use_bias)
    
    def forward(self, x):
        return self.down_size(x)

class UnetCoupleDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            attention_skip=True,
            couple_unet = False,
            path_load_checkpoint = None
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
        self.couple_unet = couple_unet
        self.path_load_checkpoint = path_load_checkpoint
        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder # (2048, 1024, 512, 256, 64)

        # computing blocks input and output channels
        head_channels = encoder_channels[0] # 2048
        in_channels = [head_channels] + list(decoder_channels[:-1]) # [2048, 256, 128, 64, 32]
        skip_channels = list(encoder_channels[1:]) + [0] # [1024, 512, 256, 64, 0]
        out_channels = decoder_channels # (256, 128, 64, 32, 16)

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, skip_from_U1=False, attention_type=attention_type)
        blocks_1 = [
            CoupleDecoderAttention(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks_1 = nn.ModuleList(blocks_1)
        if self.couple_unet:
            kwargs = dict(use_batchnorm=use_batchnorm, skip_from_U1=True, attention_type=attention_type)
            blocks_2 = [
                CoupleDecoderAttention(in_ch, skip_ch, out_ch, **kwargs)
                for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
            ]
            self.blocks_2 = nn.ModuleList(blocks_2)

            self.skip_bottom_2 = md.Conv2dReLU(
                            2048 + 2048, 2048,
                            kernel_size=3,
                            padding=1,
                            use_batchnorm=use_batchnorm,
                        )
        if self.path_load_checkpoint and self.couple_unet:
            state_dict_decoder = OrderedDict()
            pretrained_backbone = torch.load(self.path_load_checkpoint)
            for key, value in pretrained_backbone.items():
                if "decoder" in key:
                    key_encoder = key.replace("module.decoder.blocks_1.", "")
                    state_dict_decoder[key_encoder] = value
            self.blocks_1.load_state_dict(state_dict_decoder, strict=True)
            for params in self.blocks_1.parameters():
                params.requires_grad = False
        if self.path_load_checkpoint == None:
            self.initialize()
    
    def initialize(self):
        initialize_module(self.blocks_1)
        if self.couple_unet:
            initialize_module(self.blocks_2)

    def forward_U1(self, features_1):
        x_1, x_2 = None, None
        features_1 = features_1[1:]    # remove first skip with same spatial resolution
        features_1 = features_1[::-1]  # reverse channels to start from head of encoder

        head_1 = features_1[0]
        skips_1 = features_1[1:]
        x_1 = self.center(head_1)
        
        decoder_1 = []
        for i, decoder_block in enumerate(self.blocks_1):
            skip = skips_1[i] if i < len(skips_1) else None
            x_1 = decoder_block(x_1, skip)
            decoder_1.append(x_1)
        return {
                "x_1": x_1,
                "x_2": None,
            }, decoder_1

    def forward_U2(self, features_1, features_2, decoder_1):
        x_1, x_2 = None, None
        features_1 = features_1[1:]    # remove first skip with same spatial resolution
        features_1 = features_1[::-1]  # reverse channels to start from head of encoder

        head_1 = features_1[0]
        skips_1 = features_1[1:]
        x_1 = self.center(head_1)

        features_2 = features_2[1:]    # remove first skip with same spatial resolution
        features_2 = features_2[::-1]  # reverse channels to start from head of encoder
        head_2 = features_2[0]
        skips_2 = features_2[1:]

        #### them skip head_1 vao head_2 
        x_2 = torch.cat([head_1, head_2], dim=1)
        x_2 = self.skip_bottom_2(x_2)
        x_2 = self.center(head_2)
        decoder_2 = []
        for i, decoder_block in enumerate(self.blocks_2):
            skip = skips_2[i] if i < len(skips_2) else None
            skip_decoder = decoder_1[i]
            x_2 = decoder_block(x_2, skip, skip_decoder)
            decoder_2.append(x_2)
        return {
                "x_1": None,
                "x_2": x_2,
            }, decoder_2

