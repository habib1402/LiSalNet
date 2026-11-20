"""MobileNetV2 feature encoder for LiSalNet, ending at feature index 17.

The classification-only 320->1280 terminal projection is not registered or run.
Feature indices and state-dict names of all retained layers are unchanged.

The encoder structure follows torchvision's MobileNetV2 implementation, which
is distributed under the BSD 3-Clause license. See THIRD_PARTY_NOTICES.md.
"""
from torch import nn
from torch.hub import load_state_dict_from_url


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1):
        padding = (kernel_size - 1) // 2
        if dilation != 1:
            padding = dilation
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, dilation=dilation),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=None, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        inverted_residual_setting = [
            # t, c, n, s, d
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, 2, 1],
            [6, 96, 3, 1, 1],
            [6, 160, 3, 2, 1],
            [6, 320, 1, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s, d in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                dilation = d if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, dilation=d))
                input_channel = output_channel
        # End at index 17: its 320-channel output is the final saliency feature.
        # The original index-18 Conv/BN/ReLU6 block is deliberately omitted.
        # make it nn.Sequential
        self.features = nn.Sequential(*features)


        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        res = []
        for idx, m in enumerate(self.features):
            x = m(x)
            if idx in [1,3,6,13,17]:
                res.append(x)
        return res


def mobilenet_v2(pretrained=True, progress=True, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        print("loading imagenet pretrained mobilenetv2")
        # Official ImageNet weights also contain the omitted terminal block and
        # classifier. Discard only these exact known keys; validate everything else.
        omitted = {
            'features.18.0.weight', 'features.18.1.weight',
            'features.18.1.bias', 'features.18.1.running_mean',
            'features.18.1.running_var', 'features.18.1.num_batches_tracked',
            'classifier.1.weight', 'classifier.1.bias',
        }
        state_dict = {k: v for k, v in state_dict.items() if k not in omitted}
        # Older official ImageNet checkpoints omit this non-learned BN counter.
        for key, value in model.state_dict().items():
            if key.endswith('.num_batches_tracked') and key not in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict, strict=True)
        print("loaded imagenet pretrained mobilenetv2")
    return model
