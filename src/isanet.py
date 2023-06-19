import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context

import numpy as np

from src.isablock import ISA_Module
from src.backbone import ResNet


class ISANet(nn.Cell): 
    def __init__(self, num_classes):
        super(ISANet, self).__init__()

        self.backbone = ResNet.resnet50(replace_stride_with_dilation=[1,2,4])
        self.ISAHead = nn.SequentialCell(
            nn.Conv2d(2048, 512, 3, padding=1, stride=1, pad_mode="pad"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ISA_Module(in_channels=512, key_channels=256, value_channels=512, out_channels=512, dropout=0) # TODO: add dropout.
        ) 
        self.cls_head = nn.Conv2d(512, num_classes, 1, padding=0, stride=1, pad_mode="same", has_bias=True)

    def construct(self, x):

        # TODO: add aux head from backbone C3 input.

        x = self.backbone(x) # (1, 2048, 64, 128)
        x = self.ISAHead(x) # (1, 512, 64, 128)
        x = self.cls_head(x) # (1, 19, 64, 128)
        x = ops.interpolate(x, mode="bilinear", scales=(1.,1.,8.,8.)) # (1, 19, 512, 1024) 

        return x 

if __name__ == '__main__':
    
    context.set_context(mode=context.GRAPH_MODE)

    net = ISANet(19)
    x = Tensor(np.random.rand(1, 3, 512, 1024), dtype=ms.float32)
    y = net(x)

    print(y.shape) # (1, 19, 512, 1024)
