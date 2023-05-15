'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
# import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        self.layers = self._make_layers(in_planes=32)
        
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.relu2 = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(4)
        
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.relu2(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def mobilenetv2(num_classes=10):
    return MobileNetV2(num_classes=num_classes)

def test():
    net = mobilenetv2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

def save_model_to_onnx():
    import onnx
    data = torch.rand((1, 3, 32, 32)).cuda()

    net = mobilenetv2().cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar10_mobilenetv2.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path) 

    net = mobilenetv2(num_classes=100).cuda()
    onnx_path = '/data/gxy/legodnn-public-version_9.27/cv_task/image_classification/cifar/onnx_model_weight/cifar100_mobilenetv2.onnx'
    torch.onnx.export(net, data, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path) 

if __name__=='__main__':
    
    test()
    save_model_to_onnx()

    from legodnn.block_detection.model_topology_extraction import topology_extraction
    net = mobilenetv2().cuda()
    print(net)
    graph = topology_extraction(net, (1,3,32,32))
    graph.print_order_node()

