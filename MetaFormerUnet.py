import torch
import torch.nn as nn
from timm import create_model
import torch.nn.functional as F
from typing import Optional, List
from timm.layers import ClassifierHead
from metaformer import MetaFormer, caformer_s18

import torch
import torch.nn as nn
from typing import List
from functools import partial

class SegmentationHead(nn.Module):
    """
    Segmentation head which applies the final conv layer to the decoder output
    to get the required number of segmentation classes.
    """

    def __init__(self, in_channels, out_channels, activation=None):
        super(SegmentationHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class UNetWithMetaFormer(nn.Module):
    def __init__(
            self,
            backbone='MetaFormer',
            encoder_freeze=False,
            pretrained=True,
            weights=None,
            preprocessing=False,
            non_trainable_layers=(0, 1, 2, 3, 4),
            backbone_kwargs=None,
            backbone_indices=None,
            decoder_use_batchnorm=True,
            decoder_channels=(512, 320, 128, 64),
            in_channels=3,
            num_classes=2,  # Output classes for segmentation
            center=False,
            norm_layer=nn.BatchNorm2d,
            activation=nn.ReLU
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}

        # Initialize MetaFormer backbone
        self.encoder = caformer_s18(num_classes=num_classes)

        # Extract encoder channels from MetaFormer output dims
        encoder_channels = [64, 128, 320, 512]  # These should match MetaFormer's output dims

        # If freezing the encoder
        if encoder_freeze:
            self._freeze_encoder(non_trainable_layers)

        # UNet-like decoder
        if not decoder_use_batchnorm:
            norm_layer = None

        # Define the decoder with adjusted channels and skip connections
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=num_classes,
            norm_layer=norm_layer,
            center=center,
            activation=activation
        )

        # Final segmentation head
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            activation=activation() if activation else None
        )

        if weights:
            self.encoder.load_pretrained_weights(weights)

    def forward(self, x: torch.Tensor):
        if self.mean and self.std:
            x = self._preprocess_input(x)

        # Extract features from MetaFormer (used for skip connections)
        _, features = self.encoder.forward_features(x)  # Get features from MetaFormer stages

        # print shape of features
        for feature in features:
            print(feature.shape)

        # Reverse the feature maps for the decoder (as in U-Net)
        features = list(reversed(features))

        # Pass through the decoder
        x = self.decoder(features)

        # Apply final segmentation head
        seg_output = self.segmentation_head(x)

        return seg_output

    def _freeze_encoder(self, non_trainable_layer_idxs):
        """
        Freeze selected layers in the encoder, excluding normalization layers.
        """
        for layer_idx in non_trainable_layer_idxs:
            for param in self.encoder.stages[layer_idx].parameters():
                param.requires_grad = False

    def _preprocess_input(self, x, input_range=[0, 1], inplace=False):
        """
        Preprocess input by normalizing it with mean and std.
        """
        if not x.is_floating_point():
            raise TypeError(f"Input tensor should be a float tensor. Got {x.dtype}.")
        if x.ndim < 3:
            raise ValueError(f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {x.size()}")

        if not inplace:
            x = x.clone()

        dtype = x.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=x.device)

        if (std == 0).any():
            raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")

        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        return x.sub_(mean).div_(std)


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels=(512, 320, 128, 64),  # Matching encoder channels for symmetry
            final_channels=1,
            norm_layer=nn.BatchNorm2d,
            center=True,
            activation=nn.ReLU
    ):
        super().__init__()

        # Center block to process the deepest encoder feature map
        if center:
            channels = encoder_channels[0]
            self.center = DecoderBlock(
                channels, channels, scale_factor=1.0, activation=activation, norm_layer=norm_layer
            )
        else:
            self.center = nn.Identity()

        # Decoder blocks, reverse order of channels for upsampling
        in_channels = [enc_ch + dec_ch for enc_ch, dec_ch in zip(encoder_channels[1:], decoder_channels[:-1])]
        in_channels.insert(0, encoder_channels[0])  # Add encoder head channels as the first input
        out_channels = decoder_channels

        # Create decoder blocks
        self.blocks = nn.ModuleList()
        for in_ch, out_ch in zip(in_channels, out_channels):
            self.blocks.append(DecoderBlock(in_ch, out_ch, scale_factor=2.0, norm_layer=norm_layer))


    def forward(self, features: List[torch.Tensor]):
        # Reverse order to process from deepest encoder layer to shallowest
        x = self.center(features[0])
        skips = features[0:]

        # Process through decoder blocks and concatenate skip connections
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            if skip is not None:
                # Concatenate the skip connection from the encoder
                x = torch.cat([x, skip], dim=1)

            # Pass through the decoder block
            x = block(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2.0, norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.activation = activation()

        # Transpose convolution (deconvolution) for upsampling
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = norm_layer(out_channels) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        # Upsample the feature map using ConvTranspose2d
        x = self.upsample(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x

class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, activation=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = norm_layer(out_channels)
        self.act = activation(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


