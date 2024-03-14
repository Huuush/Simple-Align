# pylint: skip-file
#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from collections import OrderedDict



@ARCH_REGISTRY.register()
class QuickSRNetBase(nn.Module):
    """
    Base class for all QuickSRNet variants.

    Note on supported scaling factors: this class supports integer scaling factors. 1.5x upscaling is
    the only non-integer scaling factor supported.
    """

    def __init__(self,
                 scaling_factor,
                 num_channels,
                 num_intermediate_layers,
                 use_ito_connection,
                #  mode,
                 in_channels=3,
                 out_channels=3):
        """
        :param scaling_factor:           scaling factor for LR-to-HR upscaling (2x, 3x, 4x... or 1.5x)
        :param num_channels:             number of feature channels for convolutional layers
        :param num_intermediate_layers:  number of intermediate conv layers
        :param use_ito_connection:       whether to use an input-to-output residual connection or not
                                         (using one facilitates quantization)
        :param in_channels:              number of channels for LR input (default 3 for RGB frames)
        :param out_channels:             number of channels for HR output (default 3 for RGB frames)
        """

        super().__init__()
        self.out_channels = out_channels
        self._use_ito_connection = use_ito_connection
        self._has_integer_scaling_factor = float(scaling_factor).is_integer()
        # self.mode = mode

        if self._has_integer_scaling_factor:
            self.scaling_factor = int(scaling_factor)

        elif scaling_factor == 1.5:
            self.scaling_factor = scaling_factor

        else:
            raise NotImplementedError(f'1.5 is the only supported non-integer scaling factor. '
                                      f'Received {scaling_factor}.')

        intermediate_layers = []
        for _ in range(num_intermediate_layers):
            intermediate_layers.extend([
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
                nn.Hardtanh(min_val=0., max_val=1.)
            ])

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
            nn.Hardtanh(min_val=0., max_val=1.),
            *intermediate_layers,
        )

        if scaling_factor == 1.5:
            cl_in_channels = num_channels * (2 ** 2)
            cl_out_channels = out_channels * (3 ** 2)
            cl_kernel_size = (1, 1)
            cl_padding = 0
        else:
            cl_in_channels = num_channels
            cl_out_channels = out_channels * (self.scaling_factor ** 2)
            cl_kernel_size = (3, 3)
            cl_padding = 1

        self.conv_last = nn.Conv2d(in_channels=cl_in_channels, out_channels=cl_out_channels, kernel_size=cl_kernel_size, padding=cl_padding)

        if use_ito_connection:
            self.add_op = AddOp()

            if scaling_factor == 1.5:
                self.anchor = AnchorOp(scaling_factor=3, kernel_size=3, stride=2, padding=1,
                                              freeze_weights=False)
            else:
                self.anchor = AnchorOp(scaling_factor=self.scaling_factor,
                                              freeze_weights=False)


        if scaling_factor == 1.5:
            self.space_to_depth = nn.PixelUnshuffle(2)
            self.depth_to_space = nn.PixelShuffle(3)
        else:
            self.depth_to_space = nn.PixelShuffle(self.scaling_factor)

        self.clip_output = nn.Hardtanh(min_val=0., max_val=1.)

        self.initialize()
        
        self._is_dcr = False

    def forward(self, input):
        x = self.cnn(input)

        if not self._has_integer_scaling_factor:
            x = self.space_to_depth(x)

        if self._use_ito_connection:
            residual = self.conv_last(x)
            input_convolved = self.anchor(input)
            x = self.add_op(input_convolved, residual)
        else:
            self.out = x
            x = self.conv_last(x)

        x = self.clip_output(x)

        return self.depth_to_space(x)
    
    # def to_dcr(self):
    #     if not self._is_dcr:
    #         if self.scaling_factor == 1.5:
    #             self.conv_last = convert_conv_following_space_to_depth_to_dcr(self.conv_last, 2)
    #             self.conv_last = convert_conv_preceding_depth_to_space_to_dcr(self.conv_last, 3)
    #             if self._use_ito_connection:
    #                 self.anchor = convert_conv_preceding_depth_to_space_to_dcr(self.anchor, 3)
    #         else:
    #             self.conv_last = convert_conv_preceding_depth_to_space_to_dcr(self.conv_last, self.scaling_factor)
    #             if self._use_ito_connection:
    #                 self.anchor = convert_conv_preceding_depth_to_space_to_dcr(self.anchor, self.scaling_factor)
    #         self._is_dcr = True

    def initialize(self):
        for conv_layer in self.cnn:
            # Initialise each conv layer so that it behaves similarly to: 
            # y = conv(x) + x after initialization
            if isinstance(conv_layer, nn.Conv2d):
                middle = conv_layer.kernel_size[0] // 2
                num_residual_channels = min(conv_layer.in_channels, conv_layer.out_channels)
                with torch.no_grad():
                    for idx in range(num_residual_channels):
                        conv_layer.weight[idx, idx, middle, middle] += 1.

        if not self._use_ito_connection:
            # This will initialize the weights of the last conv so that it behaves like:
            # y = conv(x) + repeat_interleave(x, scaling_factor ** 2) after initialization
            middle = self.conv_last.kernel_size[0] // 2
            out_channels = self.conv_last.out_channels
            scaling_factor_squarred = out_channels // self.out_channels
            with torch.no_grad():
                for idx_out in range(out_channels):
                    idx_in = (idx_out % out_channels) // scaling_factor_squarred
                    self.conv_last.weight[idx_out, idx_in, middle, middle] += 1.

@ARCH_REGISTRY.register()
class QuickSRNetSmall(QuickSRNetBase):

    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=32,
            num_intermediate_layers=2,
            use_ito_connection=False,
            **kwargs
        )

@ARCH_REGISTRY.register()
class QuickSRNetMedium(QuickSRNetBase):

    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=32,
            num_intermediate_layers=5,
            use_ito_connection=False,
            **kwargs
        )

@ARCH_REGISTRY.register()
class QuickSRNetLarge(QuickSRNetBase):

    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=64,
            num_intermediate_layers=11,
            use_ito_connection=True,
            **kwargs
        )

class AnchorOp(nn.Module):
    """
    Repeat interleaves the input scaling_factor**2 number of times along the channel axis.
    """
    def __init__(self, scaling_factor, in_channels=3, init_weights=True, freeze_weights=True, kernel_size=1, **kwargs):
        """
        Args:
            scaling_factor: Scaling factor
            init_weights:   Initializes weights to perform nearest upsampling (Default for Anchor)
            freeze_weights:         Whether to freeze weights (if initialised as nearest upsampling weights)
        """
        super().__init__()

        self.net = nn.Conv2d(in_channels=in_channels,
                             out_channels=(in_channels * scaling_factor**2),
                             kernel_size=kernel_size,
                             **kwargs)

        if init_weights:
            num_channels_per_group = in_channels // self.net.groups
            weight = torch.zeros(in_channels * scaling_factor**2, num_channels_per_group, kernel_size, kernel_size)

            bias = torch.zeros(weight.shape[0])
            for ii in range(in_channels):
                weight[ii * scaling_factor**2: (ii + 1) * scaling_factor**2, ii % num_channels_per_group,
                kernel_size // 2, kernel_size // 2] = 1.

            new_state_dict = OrderedDict({'weight': weight, 'bias': bias})
            self.net.load_state_dict(new_state_dict)

            if freeze_weights:
                for param in self.net.parameters():
                    param.requires_grad = False

    def forward(self, input):
        return self.net(input)
    
class AddOp(nn.Module):
    def forward(self, x1, x2):
        return x1 + x2