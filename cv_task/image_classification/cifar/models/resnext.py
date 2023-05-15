'''ResNeXt in PyTorch.

See the paper "Aggregated Residual Transformations for Deep Neural Networks" for more details.
'''
import torch
import torch.nn as nn
# import torch.nn.functional as F


class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        # self.layer4 = self._make_layer(num_blocks[3], 2)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=8)
        self.linear = nn.Linear(cardinality*bottleneck_width*8, num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnext29_2x64d(num_classes=10):
    return ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64, num_classes=num_classes)

def resnext29_4x64d(num_classes=10):
    return ResNeXt(num_blocks=[3,3,3], cardinality=4, bottleneck_width=64, num_classes=num_classes)

def resnext29_8x64d(num_classes=10):
    return ResNeXt(num_blocks=[3,3,3], cardinality=8, bottleneck_width=64, num_classes=num_classes)

def resnext29_32x4d(num_classes=10):
    return ResNeXt(num_blocks=[3,3,3], cardinality=32, bottleneck_width=4, num_classes=num_classes)

# def resnext29_64x2d(num_classes=10):
#     return ResNeXt(num_blocks=[3,3,3], cardinality=64, bottleneck_width=2, num_classes=num_classes)

def save_model_to_onnx():
    import onnx
    data = torch.rand((1, 3, 32, 32)).cuda()

    # resnet18
    net = resnext29_2x64d().cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar10_resnext29_2x64d.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    net = resnext29_4x64d().cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar10_resnext29_4x64d.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    net = resnext29_8x64d().cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar10_resnext29_8x64d.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    net = resnext29_32x4d().cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar10_resnext29_32x4d.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    # resnet18
    net = resnext29_2x64d(num_classes=100).cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar100_resnext29_2x64d.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    net = resnext29_4x64d(num_classes=100).cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar100_resnext29_4x64d.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    net = resnext29_8x64d(num_classes=100).cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar100_resnext29_8x64d.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    net = resnext29_32x4d(num_classes=100).cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar100_resnext29_32x4d.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

if __name__=='__main__':

    save_model_to_onnx()

    from legodnn.block_detection.model_topology_extraction import topology_extraction
    net = resnext29_2x64d().cuda()
    print(net)
    graph = topology_extraction(net, (1,3,32,32))
    graph.print_order_node()