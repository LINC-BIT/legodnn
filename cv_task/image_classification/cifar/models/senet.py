'''SENet in PyTorch.

SENet is the winner of ImageNet-2017. The paper is not released yet.
'''
import torch
import torch.nn as nn
# import torch.nn.functional as F

class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.avg_pool2d = nn.AvgPool2d(kernel_size=4)
        self.relu3 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.conv2(self.relu2(self.bn2(out)))

        # Squeeze
        # w = F.avg_pool2d(out, int(out.size(2)))
        # w = F.relu(self.fc1(w))
        # w = F.sigmoid(self.fc2(w))
        self.avg_pool2d.kernel_size = int(out.size(2))
        w = self.avg_pool2d(out)
        w = self.relu3(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class SENet(nn.Module):
    def __init__(self, block=PreActBlock, num_blocks=[2,2,2,2], num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=4)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
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


def senet18(num_classes=10):
    return SENet(PreActBlock, [2,2,2,2], num_classes=num_classes)

def test():
    net = senet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
def save_model_to_onnx():
    import onnx
    data = torch.rand((1, 3, 32, 32)).cuda()

    net = senet18().cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar10_senet18.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)


    net = senet18(num_classes=100).cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar100_senet18.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

if __name__=='__main__':
    save_model_to_onnx()
    # test()

    # from legodnn.block_detection.model_topology_extraction import topology_extraction
    # net = senet18().cuda()
    # print(net)
    # graph = topology_extraction(net, (1,3,32,32))
    # graph.print_order_node()
