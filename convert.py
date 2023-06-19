import mindspore as ms
from mindspore import Tensor
import numpy as np
# pth to ckpt script.

import torch
from src.isanet import ISANet

# ckpt_path = "./resnet50v1d.ckpt"
# param_convert(ms_params, pt_params, ckpt_path)

if __name__ == '__main__':
    
    # params_torch = torch.load(pth_path)
    # for param in params_torch:
    #     print(param, params_torch[param].shape)

    ms.set_context(device_target="CPU")

    pth_path = "/home/mindocr/w30005666/isanet/resnetv1d50_b32x8_imagenet_20210531-db14775a.pth"
    save_path = "/home/mindocr/w30005666/isanet/resnetv1d50.ckpt"
    empty_ckpt_path = "/home/mindocr/w30005666/isanet/empty.ckpt"
    net_param_path = "/home/mindocr/w30005666/isanet/isanet_params.txt"

    # # create empty ckpt
    # empty_param = [{"name":"backbone.conv1.weight", "data":Tensor(np.random.rand(32,3,3,3))}]
    # ms.save_checkpoint(empty_param, empty_ckpt_path)

    # ms_ckpt = ms.load_checkpoint(ckpt_path)

    # # print params key to load.
    # file = open(net_param_path)
    # line = file.readline()
    # net_keys = []
    # while line:
    #     key = str(line)
    #     net_keys.append(key)
    #     # print(key)
    #     line = file.readline()
    # file.close()

    pth_dict = torch.load(pth_path, map_location='cpu')
    ckpt_dict = OrderedDict()
    for k, v in pth_dict.items():
        name = k
        ckpt_dict[name] = Parameter(v.detach().float().numpy())
    model = ISANet()
    load_param_into_net(model, ckpt_dict, strict_load=True)
    save_checkpoint(model, save_path)

    # index = 0
    # params_to_save = []
    # for p in ms_ckpt:
    #     if "tracked" in p:
    #         continue
    #     # print(p, net_keys[index], type(ms_ckpt[p]))
    #     new_p = {"name":net_keys[index], "data":Tensor(ms_ckpt[p])}
    #     params_to_save.append(new_p)
    #     index += 1
    # print(index)
    # ms.save_checkpoint(params_to_save, new_ckpt_path)

    # for p in ms_ckpt:
    #     if "running" in p or "tracked" in p:
    #         continue
    #     print(p, ms_ckpt[p].shape)

    # params_list = []
    # num = 0
    # for p in ms_ckpt:
    #     if "track" in p:
    #         continue
    #     print(p)
    #     num += 1
    # print(num)
    #     p_new = p
    #     if "conv" in p and "stem" not in p:
    #         p_new = p_new.replace("0.conv", "conv1")
    #         p_new = p_new.replace("1.conv", "conv2")
    #         p_new = p_new.replace("2.conv", "conv3")
    #     else:
    #         p_new = p_new.replace("weight", "gamma")
    #         p_new = p_new.replace("bias", "beta")
    #     p_new = p_new.replace("stem.", "")
    #     p_new = p_new.replace("0.bn", "bn1")
    #     p_new = p_new.replace("1.bn", "bn2")
    #     p_new = p_new.replace("2.bn", "bn3")
    #     p_new = p_new.replace("running_mean", "moving_mean")
    #     p_new = p_new.replace("running_var", "moving_variance")
    #     print(p_new)
    #     print("########################################")
        # params_list.append({p_new:ms_ckpt[p]})
        # print(p)




