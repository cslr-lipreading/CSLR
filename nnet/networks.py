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

# NeuralNets
from nnet import preprocessing
from nnet import layers
from nnet import modules
from nnet import blocks
from nnet import attentions
from nnet import transforms
from nnet import normalizations
from nnet.lipreading.models.swish import Swish


###############################################################################
# Networks
###############################################################################

def channel_shuffle(x, groups: int):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            stride: int
    ) -> None:
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
            i: int,
            o: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(
            self,
            stages_repeats,
            stages_out_channels,
            num_classes: int = 128,
            inverted_residual=InvertedResidual
    ) -> None:
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
        #     nn.BatchNorm2d(output_channels),
        #     nn.ReLU(inplace=True),
        # )
        input_channels = output_channels

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # x = self.conv1(x)
        # x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet(nn.Module):
    """ ResNet (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)

    Models: 224 x 224
    ResNet18: 11,689,512 Params
    ResNet34: 21,797,672 Params
    ResNet50: 25,557,032 Params
    ResNet101: 44,549,160 Params
    Resnet152: 60,192,808 Params

    Reference: "Deep Residual Learning for Image Recognition" by He et al.
    https://arxiv.org/abs/1512.03385

    """

    def __init__(self, dim_input=3, dim_output=1000, model="ResNet50", include_stem=True, include_head=True):
        super(ResNet, self).__init__()

        assert model in ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]

        if model == "ResNet18":
            dim_stem = 64
            dim_blocks = [64, 128, 256, 512]
            num_blocks = [2, 2, 2, 2]
            bottleneck = False
        elif model == "ResNet34":
            dim_stem = 64
            dim_blocks = [64, 128, 256, 512]
            num_blocks = [3, 4, 6, 3]
            bottleneck = False
        elif model == "ResNet50":
            dim_stem = 64
            dim_blocks = [256, 512, 1024, 2048]
            num_blocks = [3, 4, 6, 3]
            bottleneck = True
        elif model == "ResNet101":
            dim_stem = 64
            dim_blocks = [256, 512, 1024, 2048]
            num_blocks = [3, 4, 23, 3]
            bottleneck = True
        elif model == "ResNet152":
            dim_stem = 64
            dim_blocks = [256, 512, 1024, 2048]
            num_blocks = [3, 8, 36, 3]
            bottleneck = True

        self.stem = nn.Sequential(
            layers.Conv2d(in_channels=dim_input, out_channels=dim_stem, kernel_size=(7, 7), stride=(2, 2),
                          weight_init="he_normal", bias=False),
            normalizations.BatchNorm2d(num_features=dim_stem),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ) if include_stem else nn.Identity()

        # Blocks
        self.blocks = nn.ModuleList()
        for stage_id in range(4):

            for block_id in range(num_blocks[stage_id]):

                # Projection Block
                if block_id == 0:
                    if stage_id == 0:
                        stride = (1, 1)
                        bottleneck_ratio = 1
                        in_features = dim_stem
                    else:
                        stride = (2, 2)
                        bottleneck_ratio = 2
                        in_features = dim_blocks[stage_id - 1]
                # Default Block
                else:
                    stride = (1, 1)
                    in_features = dim_blocks[stage_id]
                    bottleneck_ratio = 4

                if bottleneck:
                    self.blocks.append(blocks.ResNetBottleneckBlock(
                        in_features=in_features,
                        out_features=dim_blocks[stage_id],
                        bottleneck_ratio=bottleneck_ratio,
                        kernel_size=(3, 3),
                        stride=stride,
                        act_fun="ReLU",
                        joined_post_act=True
                    ))
                else:
                    self.blocks.append(blocks.ResNetBlock(
                        in_features=in_features,
                        out_features=dim_blocks[stage_id],
                        kernel_size=(3, 3),
                        stride=stride,
                        act_fun="ReLU",
                        joined_post_act=True
                    ))

        # Head
        self.head = nn.Sequential(
            layers.GlobalAvgPool2d(),
            layers.Linear(in_features=dim_blocks[-1], out_features=dim_output, weight_init="he_normal",
                          bias_init="zeros")
        ) if include_head else nn.Identity()

    def forward(self, x):

        # (B, Din, H, W) -> (B, D0, H//4, W//4)
        x = self.stem(x)

        # (B, D0, H//4, W//4) -> (B, D4, H//32, W//32)
        for block in self.blocks:
            x = block(x)

        # (B, D4, H//32, W//32) -> (B, Dout)
        x = self.head(x)

        return x


class Transformer(nn.Module):

    def __init__(self, dim_model, num_blocks, att_params={"class": "MultiHeadAttention",
                                                          "params": {"num_heads": 4, "weight_init": "normal_02",
                                                                     "bias_init": "zeros"}}, ff_ratio=4,
                 emb_drop_rate=0.1, drop_rate=0.1, act_fun="GELU", pos_embedding=None, mask=None, inner_dropout=False,
                 weight_init="normal_02", bias_init="zeros", post_norm=False):
        super(Transformer, self).__init__()

        # Positional Embedding
        self.pos_embedding = pos_embedding

        # Input Dropout
        self.dropout = nn.Dropout(p=emb_drop_rate)

        # Mask
        self.mask = mask

        # Transformer Blocks
        self.blocks = nn.ModuleList([blocks.TransformerBlock(
            dim_model=dim_model,
            ff_ratio=ff_ratio,
            att_params=att_params,
            drop_rate=drop_rate,
            inner_dropout=inner_dropout,
            act_fun=act_fun,
            weight_init=weight_init,
            bias_init=bias_init,
            post_norm=post_norm
        ) for block_id in range(num_blocks)])

        # LayerNorm
        self.layernorm = nn.LayerNorm(normalized_shape=dim_model) if not post_norm else nn.Identity()

    def forward(self, x, lengths=None):

        # Pos Embedding
        if self.pos_embedding != None:
            x = self.pos_embedding(x)

        # Input Dropout
        x = self.dropout(x)

        # Mask (1 or B, 1, N, N)
        if self.mask != None:
            mask = self.mask(x, lengths)
        else:
            mask = None

        # Transformer Blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # LayerNorm
        x = self.layernorm(x)

        return x






class VisualEfficientEncoder(nn.Module):

    def __init__(self, include_head=True, vocab_size=256, interctc_blocks=[],  # num_blocks=[6, 6]
                 loss_prefix="ctc", my_settings=None):
        super(VisualEfficientEncoder, self).__init__()

        self.my_settings = my_settings

        dim_model = self.my_settings['dim_model']
        num_blocks = self.my_settings['num_blocks']
        ch = 24
        front = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], num_classes=dim_model[0])

        self.front_end = nn.Sequential(
            modules.ConvNeuralNetwork(
                dim_input=1,
                dim_layers=ch,
                kernel_size=(5, 7, 7),
                strides=(1, 2, 2),
                norm="BatchNorm3d",
                act_fun="ReLU",
                drop_rate=0.0,
                dim=3
            ),
            layers.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding="same"),
            transforms.VideoToImages(),  # (B, C, T, H, W) -> (BT, C, H, W)
            front
        )

        self.expand_time = transforms.ImagesToVideos()


        print('gru')
        class GRU(nn.Module):
            def __init__(self):
                super(GRU, self).__init__()
                self.gru = nn.GRU(input_size=dim_model[-1], hidden_size=dim_model[-1],
                                  num_layers=num_blocks[-1], bidirectional=True, batch_first=True)
                self.linear = nn.Linear(dim_model[-1] * 2, dim_model[-1])

            def forward(self, x, l):
                x, _ = self.gru(x)
                x = self.linear(x)
                return x, l, {}
        self.back_end = GRU()

        self.head = nn.Linear(dim_model[-1], vocab_size)


    def forward(self, x, lengths):

        # Frontend
        time = x.shape[2]
        x = self.front_end(x)  # (B, C, T, H, W) -> (BT, C)
        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)  # (BT, C) -> (BT, C, 1, 1)
        x = self.expand_time(x, time)  # (BT, C, 1, 1) -> (B, C, T, 1, 1)
        x = x.squeeze(dim=-1).squeeze(dim=-1).transpose(1, 2)  # (B, C, T, 1, 1) -> (B, T, C)

        # Backend
        x, lengths, interctc_outputs = self.back_end(x, lengths)

        # Head
        x = self.head(x)

        return x, lengths, interctc_outputs
