import torch
import torch.nn as nn
from timm import create_model
import torch.nn.functional as F
from typing import Optional, List
from timm.layers import ClassifierHead
import metaformer


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
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=3,
            num_classes=2,  # Output classes for segmentation
            center=False,
            norm_layer=nn.BatchNorm2d,
            activation=nn.ReLU
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}

        # MetaFormer backbone initialization
        # encoder = MetaFormer(
        #     in_chans=in_channels,
        #     num_classes=num_classes,  # You can ignore this for segmentation
        #     pretrained_weights=weights,
        #     **backbone_kwargs
        # )

        encoder = metaformer.__dict__['caformer_s18'](num_classes=num_classes)

        # Extract channels information from MetaFormer feature maps
        encoder_channels = [64, 128, 320, 512]  # These should match MetaFormer's output dims

        self.encoder = encoder
        self.backbone = backbone
        if encoder_freeze:
            self._freeze_encoder(non_trainable_layers)

        if preprocessing:
            self.mean = [0.485, 0.456, 0.406]  # You can set this according to your dataset
            self.std = [0.229, 0.224, 0.225]
        else:
            self.mean = None
            self.std = None

        # Define the decoder using a Unet-like structure
        if not decoder_use_batchnorm:
            norm_layer = None
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=num_classes,  # Set the final channels to number of segmentation classes
            norm_layer=norm_layer,
            center=center,
            activation=activation
        )

        # Segmentation head that maps final decoder output to segmentation classes
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],  # The smallest feature map size from decoder
            out_channels=num_classes,  # Number of segmentation classes
            activation=None  # Apply softmax/sigmoid as needed in the main training loop
        )

        if weights:
            self.encoder.load_state_dict(torch.load(weights), strict=True)

    def forward(self, x: torch.Tensor):
        if self.mean and self.std:
            x = self._preprocess_input(x)

        # Get encoder features from MetaFormer (used for skip connections)
        _, features = self.encoder.forward_features(x)  # Get features from different stages

        # Reverse the feature maps for the decoder
        features = list(reversed(features))

        # Decoder forward pass using the features
        x = self.decoder(features)

        # Apply segmentation head
        seg_output = self.segmentation_head(x)

        return seg_output

    @torch.no_grad()
    def predict(self, x):
        if self.training: self.eval()
        seg_output = self.forward(x)
        return seg_output

    def _freeze_encoder(self, non_trainable_layer_idxs):
        """
        Set selected layers non-trainable, excluding BatchNormalization layers.
        Parameters
        ----------
        non_trainable_layer_idxs: tuple
            Specifies which layers are non-trainable for
            list of the non-trainable layer names.
        """
        non_trainable_layers = [
            self.encoder.feature_info[layer_idx]["module"].replace(".", "_") \
            for layer_idx in non_trainable_layer_idxs
        ]
        for layer in non_trainable_layers:
            for child in self.encoder[layer].children():
                for param in child.parameters():
                    param.requires_grad = False
        return

    def _preprocess_input(self, x, input_range=[0, 1], inplace=False):
        if not x.is_floating_point():
            raise TypeError(f"Input tensor should be a float tensor. Got {x.dtype}.")

        if x.ndim < 3:
            raise ValueError(
                f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {x.size()}"
            )

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


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, scale_factor=2.0,
        activation=nn.ReLU, norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, activation=activation)
        self.scale_factor = scale_factor
        if norm_layer is None:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels,  **conv_args)
        else:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, norm_layer=norm_layer, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels, norm_layer=norm_layer, **conv_args)

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        if self.scale_factor != 1.0:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            norm_layer=nn.BatchNorm2d,
            center=True,
            activation=nn.ReLU
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = DecoderBlock(
                channels, channels, scale_factor=1.0,
                activation=activation, norm_layer=norm_layer
            )
        else:
            self.center = nn.Identity()

        # list(decoder_channels[:-1][:len(encoder_channels)])
        in_channels = [in_chs + skip_chs for in_chs, skip_chs in zip(
            [encoder_channels[0]] + list(decoder_channels[:-1]),
            list(encoder_channels[1:]) + [0])]

        out_channels = decoder_channels

        if len(in_channels) != len(out_channels):
            in_channels.append(in_channels[-1]//2)

        self.blocks = nn.ModuleList()
        for in_chs, out_chs in zip(in_channels, out_channels):
            self.blocks.append(DecoderBlock(in_chs, out_chs, norm_layer=norm_layer))
        self.final_conv = nn.Conv2d(out_channels[-1], final_channels, kernel_size=(1, 1))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: List[torch.Tensor]):
        encoder_head = x[0]
        skips = x[1:]
        x = self.center(encoder_head)
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, skip)
        x = self.final_conv(x)
        return x

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
