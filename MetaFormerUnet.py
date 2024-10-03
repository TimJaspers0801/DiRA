import torch
import torch.nn as nn
from timm import create_model
import torch.nn.functional as F
from typing import Optional, List
from timm.layers import ClassifierHead
import metaformer

class UNetWithMetaFormer(nn.Module):
    """
    U-Net with MetaFormer as the backbone (encoder).
    This implementation uses the MetaFormer architecture for feature extraction and
    integrates it with a U-Net-style decoder.
    """
    def __init__(
            self,
            backbone='metaformer',
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
            num_classes=2,
            center=False,
            norm_layer=nn.BatchNorm2d,
            activation=nn.ReLU
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}

        # Initialize MetaFormer as the encoder
        # encoder = MetaFormer(
        #     in_chans=in_channels,
        #     num_classes=num_classes,  # this is not used in feature extraction
        #     pretrained_weights=weights,
        #     **backbone_kwargs
        # )

        encoder = metaformer.__dict__[args.arch](num_classes=num_classes)

        # MetaFormer returns feature maps at multiple stages
        encoder_channels = encoder.dims[::-1]  # Reverse order for decoder input
        self.encoder = encoder
        self.backbone = backbone
        if encoder_freeze:
            self._freeze_encoder(non_trainable_layers)

        if preprocessing:
            self.mean = [0.485, 0.456, 0.406]  # Mean for ImageNet
            self.std = [0.229, 0.224, 0.225]   # Standard deviation for ImageNet
        else:
            self.mean = None
            self.std = None

        if not decoder_use_batchnorm:
            norm_layer = None

        # U-Net decoder
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=num_classes,
            norm_layer=norm_layer,
            center=center,
            activation=activation
        )

        # Classification head, if needed
        self.classification_head = ClassifierHead(encoder_channels[0], 1)

    def forward(self, x: torch.Tensor):
        if self.mean and self.std:
            x = self._preprocess_input(x)

        # Pass input through MetaFormer encoder
        x, feature_maps = self.encoder.forward_features(x)

        # Reverse the order of feature maps for U-Net decoder
        feature_maps = list(reversed(feature_maps))

        # Apply classification head to the first feature map
        cls = self.classification_head(feature_maps[0])

        # Pass through U-Net decoder
        x = self.decoder(feature_maps)
        return x, cls

    @torch.no_grad()
    def predict(self, x):
        """
        Inference method. Switch model to `eval` mode,
        call `.forward(x)` with `torch.no_grad()`
        """
        if self.training:
            self.eval()
        x, cls = self.forward(x)
        return x, cls

    def _freeze_encoder(self, non_trainable_layer_idxs):
        """
        Set selected layers non trainable, excluding BatchNormalization layers.
        """
        for layer_idx in non_trainable_layer_idxs:
            for param in self.encoder.stages[layer_idx].parameters():
                param.requires_grad = False

    def _preprocess_input(self, x, input_range=[0, 1], inplace=False):
        """
        Preprocess input according to the mean and std for the encoder.
        """
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