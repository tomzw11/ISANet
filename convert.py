# pth to ckpt script.
import mindspore as ms
from mindspore import Tensor, Parameter, load_param_into_net, save_checkpoint
import numpy as np
import torch

# pth to ckpt script.
# export LD_PRELOAD=/home/mindocr/miniconda3/envs/ms1.9_py37/lib/python3.7/site-packages/torch/lib/libgomp-d22c30c5.so.1

import sys
from src.isanet import ISANet
from src.backbone import ResNet

def read(path):

    file = open(path, "r")
    res = []
    for line in file:
        res.append(str(line).strip("\n"))
    file.close()
    return res

def mapping(p):

    if "stem" in p:
        p = p.replace("0.bn", "bn1")
        p = p.replace("1.bn", "bn2")
        p = p.replace("2.bn", "bn3")
        p = p.replace("bn1.weight", "bn1.gamma")
        p = p.replace("bn2.weight", "bn2.gamma")
        p = p.replace("bn3.weight", "bn3.gamma")
        p = p.replace("bn1.bias", "bn1.beta")
        p = p.replace("bn2.bias", "bn2.beta")
        p = p.replace("bn3.bias", "bn3.beta")
        p = p.replace("0.conv.weight", "conv1.weight")
        p = p.replace("1.conv.weight", "conv2.weight")
        p = p.replace("2.conv.weight", "conv3.weight")
        p = p.replace("stem.", "")
    elif "layer" in p:
        if "bn" in p:
            p = p.replace("weight", "gamma")
            p = p.replace("bias", "beta")
        if "downsample" in p:
            if "layer1.0" not in p:
                p = p.replace("downsample.1", "downsample.0")
                p = p.replace("downsample.2", "downsample.1")
            p = p.replace("downsample.1.weight", "downsample.1.gamma")
            p = p.replace("downsample.1.bias", "downsample.1.beta")
        p = p.replace("bias", "beta")
    else:
        p = p.replace("head", "backbone")

    p = p.replace("running_mean", "moving_mean")
    p = p.replace("running_var", "moving_variance")

    return p

if __name__ == '__main__':
    
    ms.set_context(device_target="CPU")

    pth_path = "/home/mindocr/w30005666/isanet/resnetv1d50_b32x8_imagenet_20210531-db14775a.pth"
    save_path = "/home/mindocr/w30005666/isanet/resnetv1d50.ckpt"

    pth_dict = torch.load(pth_path, map_location='cpu')
    ckpt_dict = {}
    for k, v in pth_dict["state_dict"].items():
        name = mapping(k)
        print(name)
        p = Parameter(v.detach().float().numpy())
        ckpt_dict[name] = p


    # model = ISANet(19)
    model = ResNet.resnet50(replace_stride_with_dilation=[1,2,4], pretrained_path=None)
    load_param_into_net(model, ckpt_dict, strict_load=False)
    save_checkpoint(model, save_path)



