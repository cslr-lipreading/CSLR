# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn

# Neural Nets
from nnet import modules
from nnet import layers
from nnet import activations
from nnet import normalizations

###############################################################################
# ResNet Blocks
###############################################################################

class ResNetBlock(nn.Module):

    """ ResNet Residual Block used by ResNet18 and ResNet34 networks.

    References: "Deep Residual Learning for Image Recognition", He et al.
    https://arxiv.org/abs/1512.03385

    """

    def __init__(self, in_features, out_features, kernel_size, stride, norm="BatchNorm2d", act_fun="ReLU", dim=2, channels_last=False, weight_init="he_normal", bias_init="zeros", bias=False, joined_post_act=False, padding="same"):
        super(ResNetBlock, self).__init__()

        conv = {
            1: layers.Conv1d,
            2: layers.Conv2d,
            3: layers.Conv3d
        }

        # Norm
        if isinstance(norm, dict):
            norm_params = norm["params"]
            norm = normalizations.norm_dict[norm["class"]]
        else:
            norm_params = {}
            norm = normalizations.norm_dict[norm]

        # Act fun
        if isinstance(act_fun, dict):
            act_fun_params = act_fun["params"]
            act_fun = activations.act_dict[act_fun["class"]]
        else:
            act_fun_params = {}
            act_fun = activations.act_dict[act_fun]

        # layers
        self.layers = nn.Sequential(
            conv[dim](in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=stride, channels_last=channels_last, bias=bias, weight_init=weight_init, bias_init=bias_init, padding=padding),
            norm(out_features, **norm_params, channels_last=channels_last),
            act_fun(**act_fun_params),

            conv[dim](in_channels=out_features, out_channels=out_features, kernel_size=kernel_size, channels_last=channels_last, bias=bias, weight_init=weight_init, bias_init=bias_init, padding=padding),
            norm(out_features, **norm_params, channels_last=channels_last),
            nn.Identity() if joined_post_act else act_fun(**act_fun_params)
        )

        # Joined Post Act
        self.joined_post_act = act_fun(**act_fun_params) if joined_post_act else nn.Identity()

        # Residual Block
        if torch.prod(torch.tensor(stride)) > 1 or in_features != out_features:
            self.residual = nn.Sequential(
                conv[dim](in_channels=in_features, out_channels=out_features, kernel_size=1, stride=stride, channels_last=channels_last, bias=bias, weight_init=weight_init, bias_init=bias_init),
                norm(out_features, **norm_params, channels_last=channels_last)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):

        # Forward Layers
        x = self.joined_post_act(self.layers(x) + self.residual(x))

        return x

class ResNetBottleneckBlock(nn.Module):

    """ ResNet Bottleneck Residual Block used by ResNet50, ResNet101 and ResNet152 networks.

    References: "Deep Residual Learning for Image Recognition", He et al.
    https://arxiv.org/abs/1512.03385

    """

    def __init__(self, in_features, out_features, bottleneck_ratio, kernel_size, stride, norm="BatchNorm2d", act_fun="ReLU", dim=2, channels_last=False, weight_init="he_normal", bias_init="zeros", bias=False, joined_post_act=False, padding="same"):
        super(ResNetBottleneckBlock, self).__init__()

        conv = {
            1: layers.Conv1d,
            2: layers.Conv2d,
            3: layers.Conv3d
        }

        # Norm
        if isinstance(norm, dict):
            norm_params = norm["params"]
            norm = normalizations.norm_dict[norm["class"]]
        else:
            norm_params = {}
            norm = normalizations.norm_dict[norm]

        # Act fun
        if isinstance(act_fun, dict):
            act_fun_params = act_fun["params"]
            act_fun = activations.act_dict[act_fun["class"]]
        else:
            act_fun_params = {}
            act_fun = activations.act_dict[act_fun]

        # Assert
        assert in_features % bottleneck_ratio == 0

        # layers
        self.layers = nn.Sequential(
            conv[dim](in_channels=in_features, out_channels=in_features // bottleneck_ratio, kernel_size=1, channels_last=channels_last, bias=bias, weight_init=weight_init, bias_init=bias_init),
            norm(in_features // bottleneck_ratio, **norm_params, channels_last=channels_last),
            act_fun(**act_fun_params),

            conv[dim](in_channels=in_features // bottleneck_ratio, out_channels=in_features // bottleneck_ratio, kernel_size=kernel_size, stride=stride, channels_last=channels_last, bias=bias, weight_init=weight_init, bias_init=bias_init, padding=padding),
            norm(in_features // bottleneck_ratio, **norm_params, channels_last=channels_last),
            act_fun(**act_fun_params),

            conv[dim](in_channels=in_features // bottleneck_ratio, out_channels=out_features, kernel_size=1, channels_last=channels_last, bias=bias, weight_init=weight_init, bias_init=bias_init),
            norm(out_features, **norm_params, channels_last=channels_last),
            nn.Identity() if joined_post_act else act_fun(**act_fun_params)
        )

        # Joined Post Act
        self.joined_post_act = act_fun(**act_fun_params) if joined_post_act else nn.Identity()

        # Residual Block
        if torch.prod(torch.tensor(stride)) > 1 or in_features != out_features:
            self.residual = nn.Sequential(
                conv[dim](in_channels=in_features, out_channels=out_features, kernel_size=1, stride=stride, channels_last=channels_last, bias=bias, weight_init=weight_init, bias_init=bias_init),
                norm(out_features, **norm_params, channels_last=channels_last)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):

        # Forward Layers
        x = self.joined_post_act(self.layers(x) + self.residual(x))

        return x

###############################################################################
# Transformer Blocks
###############################################################################

class TransformerBlock(nn.Module):

    def __init__(self, dim_model, att_params, ff_ratio=4, drop_rate=0.1, inner_dropout=False, act_fun="GELU", weight_init="normal_02", bias_init="zeros", post_norm=False):
        super(TransformerBlock, self).__init__()

        # Muti-Head Self-Attention Module
        self.self_att_module = modules.AttentionModule(
            dim_model=dim_model,
            att_params=att_params,
            drop_rate=drop_rate,
            residual=True
        )

        # Feed Forward Module
        self.ff_module = modules.FeedForwardModule(
            dim_model=dim_model, 
            dim_ffn=dim_model * ff_ratio, 
            drop_rate=drop_rate, 
            act_fun=act_fun,
            inner_dropout=inner_dropout,
            weight_init=weight_init,
            bias_init=bias_init
        )

        # Post Norm
        self.post_norm = nn.LayerNorm(normalized_shape=dim_model) if post_norm else nn.Identity()

    def forward(self, x, mask=None):

        # Muti-Head Self-Attention Module
        x = self.self_att_module(x, mask=mask)

        # Feed Forward Module
        x = x + self.ff_module(x)

        # Post Norm
        x = self.post_norm(x)

        return x


