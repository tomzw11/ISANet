# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import ast
import os
import sys

import numpy as np
import mindspore as ms
from mindspore import context, Model, nn
from mindspore import dataset as de
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.nn import SGD
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig

from src.cityscapes import Cityscapes
from src.isanet import ISANet
from src.config import config

set_seed(1)
de.config.set_seed(1)

class WithLossCell(nn.Cell):
    def __init__(self, net, criterion):
        super(WithLossCell, self).__init__()
        self.net = net
        self.criterion = criterion

    def construct(self, img, label):
        pred = self.net(img)
        loss = self.criterion(pred, label)
        # print("label ", label.dtype)
        # print("#########################################")
        # print(pred)
        # print(label)
        return loss


def main():

    if config.run_distribute:
        init()
        device_id = int(os.getenv("DEVICE_ID")) if config.device_target == "Ascend" else get_rank()
        device_num = int(os.getenv("RANK_SIZE")) if config.device_target == "Ascend" else get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          gradients_mean=True,
                                          device_num=device_num)
    else:
        device_id = 7
        device_num = 1

    context.set_context(mode=config.context_mode, device_target=config.device_target, device_id=config.device_id)

    data_tr = Cityscapes(config.data_path,
                         num_classes=config.num_classes,
                         multi_scale=config.multi_scale,
                         flip=config.flip,
                         ignore_label=config.ignore_label,
                         base_size=config.base_size,
                         crop_size=(config.crop_size[0], config.crop_size[1]),
                         downsample_rate=config.downsample_rate,
                         scale_factor=config.scale_factor,
                         mean=config.mean,
                         std=config.std,
                         is_train=True)

    if device_num == 1:
        dataset = de.GeneratorDataset(data_tr, column_names=["image", "label"],
                                      num_parallel_workers=config.workers,
                                      shuffle=config.shuffle)
    else:
        dataset = de.GeneratorDataset(data_tr, column_names=["image", "label"],
                                      num_parallel_workers=config.workers,
                                      shuffle=config.shuffle,
                                      num_shards=device_num, shard_id=device_id)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)

    # Create network
    net = ISANet(config.num_classes, config.pretrained)
    net.set_train(True)

    # TODO: update config params.
    lr_poly = nn.polynomial_decay_lr(
        config.lr, 
        config.end_lr, 
        config.num_iterations, 
        config.step_per_epoch, 
        config.num_epochs, 
        config.lr_power)

    data_loader = dataset.create_dict_iterator()
    # image (bs, 3, 512, 1024) 
    # label (bs, 512, 1024) 

    optim = nn.SGD(params=net.trainable_params(), learning_rate=lr_poly, weight_decay=config.weight_decay)

    net_with_loss = WithLossCell(net, nn.CrossEntropyLoss(ignore_index=config.ignore_label))
    train_net = nn.TrainOneStepCell(net_with_loss, optim)

    iteration = 0
    for epoch in range(config.num_epochs):    
        for i, data in enumerate(data_loader): 

            img = data['image']
            label = data['label']
            loss = train_net(img, label)
            iteration += 1
            print("iteration ", iteration, "loss ", loss)

if __name__ == '__main__':
    main()
