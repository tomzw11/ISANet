import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    """ 3x3 convolution """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation, pad_mode="pad",
                     padding=1, has_bias=False)

class BasicBlock(nn.Cell): # 迁移完成

    expansion: int = 4
    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1,
        base_width = 64, dilation = 1, norm_layer = None):
        
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, 
                               padding=dilation, group=groups, has_bias=False, dilation=dilation, pad_mode="pad")
        
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes ,kernel_size=3, stride=stride, 
                               padding=dilation, group=groups, has_bias=False, dilation=dilation, pad_mode="pad")
        
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Cell): # 迁移完成
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample = None,
        groups = 1, base_width = 64, dilation = 1, norm_layer = None,):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=width, kernel_size=1, stride=1, has_bias=False, pad_mode="pad")
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=stride, has_bias=False, padding=dilation, dilation=dilation, pad_mode="pad")
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=planes * self.expansion, kernel_size=1, stride=1, has_bias=False, pad_mode="pad")
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Cell): 
    def __init__(
        self,block, layers,num_classes = 1000, zero_init_residual = False, groups = 1,
        width_per_group = 64, replace_stride_with_dilation = None, norm_layer = None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 2
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = conv3x3(3, 32, stride=2)
        self.bn1 = self._norm_layer(32)
        self.conv2 = conv3x3(32, 32)
        self.bn2 = self._norm_layer(32)
        self.conv3 = conv3x3(32, 64)
        self.bn3 = self._norm_layer(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Dense(512 * block.expansion, num_classes)

        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                ms.common.initializer.HeNormal(m.weight, mode="fan_out", nonlinearity="relu") 
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                ms.common.initializer.Constant(value=1)(m.gamma)
                ms.common.initializer.Constant(value=0)(m.beta)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.cells():
                if isinstance(m, Bottleneck):
                    ms.common.initializer.Constant(value=0)(m.bn3.gamma) # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    ms.common.initializer.Constant(value=0)(m.bn2.gamma) # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride = 1,
        dilate = False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = stride
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=1, stride=stride, has_bias=False, pad_mode="pad"),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.SequentialCell(*layers)

    def _forward_impl(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
        
    def construct(self, x) :
        return self._forward_impl(x)

    def _resnet(block, layers, pretrained_path=None, **kwargs,):
        model = ResNet(block, layers, **kwargs)
        if pretrained_path is not None:
            unloaded = ms.load_param_into_net(model, ms.load_checkpoint(pretrained_path), strict_load=False)
            if unloaded:
                for p in unloaded:
                    print("resnet weights unloaded ", p)
            else:
                print("all resnet weights loaded.")
        return model
    
    def resnet50(pretrained_path=None, **kwargs):
        return ResNet._resnet(Bottleneck, [3, 4, 6, 3], pretrained_path, **kwargs)
    
    def resnet101(pretrained_path=None, **kwargs):
        return ResNet._resnet(Bottleneck, [3, 4, 23, 3], pretrained_path, **kwargs)