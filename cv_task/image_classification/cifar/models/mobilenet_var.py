"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""
import torch
import torch.nn as nn
import math

__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=False)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=False)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2Var(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.):
        super(MobileNetV2Var, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 1],  # CIFAR10: stride 2 -> 1
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 1)] # CIFAR10: stride 2 -> 1
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2_w1(num_classes=10):
    return MobileNetV2Var(num_classes=num_classes, width_mult=1)

def mobilenetv2_w2(num_classes=10):
    return MobileNetV2Var(num_classes=num_classes, width_mult=2.0)

def test():
    net = mobilenetv2_w1()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())


def save_model_to_onnx():
    import onnx
    data = torch.rand((1, 3, 32, 32)).cuda()

    net = mobilenetv2_w1().cuda()
    print(net)
    exit()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar10_mobilenetv2_w1.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path) 

    net = mobilenetv2_w1(num_classes=100).cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar100_mobilenetv2_w1.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path) 

    net = mobilenetv2_w2().cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar10_mobilenetv2_w2.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path) 

    net = mobilenetv2_w2(num_classes=100).cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar100_mobilenetv2_w2.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path) 

if __name__ == '__main__':
    test()
    # save_model_to_onnx()
    model = mobilenetv2_w2()
    print(model)