"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch
"""
import math

import torch
import torch.nn as nn


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=4):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False, padding=0)
        self.norm1 = nn.BatchNorm2d(in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=False, padding=0)
        self.norm2 = nn.BatchNorm2d(num_gates)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.norm2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(
            inp,
            oup,
            kernel_size,
            stride,
            kernel_size // 2,
            groups=inp,
            bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )


class GhostModule(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 kernel_size=1,
                 ratio=2,
                 dw_size=3,
                 stride=1,
                 relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp,
                init_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(
                hidden_dim, hidden_dim, kernel_size, stride, relu=False)
            if stride == 2 else nn.Sequential(),
            # Squeeze-and-Excite
            # SELayer(hidden_dim) if use_se else nn.Sequential(),
            ChannelGate(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, 3, stride, relu=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class PFLDInference(nn.Module):
    def __init__(self):
        super(PFLDInference, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.block2_1 = conv_bn(16, 16, 3, 1)
        # self.block2_1 = GhostBottleneck(16, 16, 16, 3, 1, 1)
        self.block2_2 = GhostBottleneck(16, 72, 24, 3, 2, 0)
        self.block2_3 = GhostBottleneck(24, 88, 24, 3, 1, 0)

        self.block3_1 = GhostBottleneck(24, 96, 40, 5, 2, 1)
        self.block3_2 = GhostBottleneck(40, 240, 40, 5, 1, 1)
        self.block3_3 = GhostBottleneck(40, 240, 40, 5, 1, 1)
        self.block3_4 = GhostBottleneck(40, 120, 48, 5, 1, 1)
        # self.block3_5 = GhostBottleneck(48, 144, 48, 5, 1, 1)
        self.block3_6 = GhostBottleneck(48, 96, 24, 5, 1, 1)  # [24, 14, 14]

        self.block4_1 = GhostBottleneck(24, 288, 96, 5, 2, 1)
        # self.block4_2 = GhostBottleneck(96, 576, 96, 5, 1, 1)
        self.block4_3 = GhostBottleneck(96, 288, 48, 5, 1, 1)  # [48, 14, 14]

        self.conv5 = nn.Conv2d(48, 96, 7, 1, 0)  # [96, 1, 1]
        self.bn5 = nn.BatchNorm2d(96)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(168, 212)

    def forward(self, x):  # x: 3, 112, 112
        x = self.relu(self.bn1(self.conv1(x)))  # [16, 56, 56]
        x = self.block2_1(x)
        x = self.block2_2(x)
        out1 = self.block2_3(x)

        x = self.block3_1(out1)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        # x = self.block3_5(x)
        x = self.block3_6(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.block4_1(x)
        # x = self.block4_2(x)
        x = self.block4_3(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv5(x))
        x3 = x3.view(x3.size(0), -1)
        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)
        landmarks = landmarks.view(-1, 106, 2)
        return out1, landmarks


class AuxiliaryNet(nn.Module):
    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = conv_bn(24, 96, 3, 2)
        self.conv2 = conv_bn(96, 96, 3, 1)
        self.conv3 = conv_bn(96, 32, 3, 2)
        self.conv4 = conv_bn(32, 128, 7, 1)
        self.max_pool1 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class GhostPFLDNet(nn.Module):
    def __init__(self, pretrained=False):
        super(GhostPFLDNet, self).__init__()
        self.pdfl_inference = PFLDInference()
        self.auxiliary_net = AuxiliaryNet()
        if pretrained:
            self.init_weight()

    def forward(self, x):
        feature, landmarks = self.pdfl_inference(x)
        angle = self.auxiliary_net(feature)
        return angle, landmarks

    def init_weight(self):
        def load_checkpoint(model, checkpoint):
            key_list = []
            discarded_layers = []
            mapped_state_dict = model.state_dict()
            for key, value in mapped_state_dict.items():
                key_list.append(key)
            for index, (key, value) in enumerate(checkpoint.items()):
                if value.size() != mapped_state_dict[key_list[index]].size():
                    discarded_layers.append(key_list[index])
                    continue
                # print("load ", index, key, key_list[index])
                mapped_state_dict[key_list[index]] = value
            model.load_state_dict(mapped_state_dict)
            if len(discarded_layers) > 0:
                print('** The following layers are discarded '
                      'due to unmatched layer size: {}'.format(discarded_layers))

        model_path = r"checkpoints/pretrained/checkpoint.pth.tar"
        checkpoint = torch.load(model_path, map_location="cpu")
        load_checkpoint(self.pdfl_inference, checkpoint['plfd_backbone'])
        load_checkpoint(self.auxiliary_net, checkpoint['auxiliarynet'])


if __name__ == '__main__':
    input = torch.randn(1, 3, 112, 112)
    plfd_backbone = PFLDInference()
    auxiliarynet = AuxiliaryNet()
    features, landmarks = plfd_backbone(input)
    angle = auxiliarynet(features)

    print("angle.shape:{0:}, landmarks.shape: {1:}".format(
        angle.shape, landmarks.shape))
