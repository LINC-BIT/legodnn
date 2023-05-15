'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
from torch._C import set_flush_denormal
import torch.nn as nn
# import torch.nn.functional as F
import torchvision.models.resnet 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.relu2 = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu3 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=4)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def save_model_to_onnx():
    import onnx
    data = torch.rand((1, 3, 32, 32)).cuda()

    # resnet18
    net = resnet18().cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar10_resnet18.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    # resnet34
    net = resnet34().cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar10_resnet34.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    # resnet34
    net = resnet50().cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar10_resnet50.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    # resnet34
    net = resnet101().cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar10_resnet101.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    # resnet34
    net = resnet152().cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar10_resnet152.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    # resnet18
    net = resnet18(num_classes=100).cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar100_resnet18.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    # resnet34
    net = resnet34(num_classes=100).cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar100_resnet34.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    # resnet34
    net = resnet50(num_classes=100).cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar100_resnet50.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    # resnet34
    net = resnet101(num_classes=100).cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar100_resnet101.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

    # resnet34
    net = resnet152(num_classes=100).cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar100_resnet152.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

if __name__=='__main__':

    save_model_to_onnx()

    from legodnn.block_detection.model_topology_extraction import topology_extraction
    net = resnet18().cuda()
    print(net)
    graph = topology_extraction(net, (1,3,32,32))
    graph.print_order_node()
