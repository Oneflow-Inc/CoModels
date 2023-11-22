# Copyright (c) 2023 OneFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import oneflow as flow
from .utils import cfg


class FC(flow.nn.Module):

    def __init__(self, num_ins, num_outs, num_layers, hiden_size, activiation):
        super(FC, self).__init__()
        if num_ins <= 0 or num_outs <= 0 or hiden_size <= 0 or num_layers <= 2:
            raise ValueError(
                'Invalid value, num_ins/num_outs/hiden_size should be greater than 0 and num_layers should be greater than 2.'
            )
        self.num_ins = num_ins
        self.num_outs = num_outs
        self.num_layers = num_layers
        self.hiden_size = hiden_size
        self.activiation = activiation

        layers = [flow.nn.Linear(num_ins, hiden_size), self.activiation]
        for idx in range(num_layers - 2):
            layers.append(flow.nn.Linear(hiden_size, hiden_size))
            layers.append(self.activiation)
        layers.append(flow.nn.Linear(hiden_size, num_outs))
        self.layers = flow.nn.Sequential(*layers)
        self.to(cfg.get_device())

    def forward(self, ins):
        return self.layers(ins)


class Loss(object):

    def __init__(self, name):
        self.name = name

    def evaluate(self, items):
        raise NotImplementedError("Implement in Loss subclass")


class WeightedL2(Loss):

    def __init__(self, weights):
        super(WeightedL2, self).__init__('WeightedL2')
        self.weights = weights

    def evaluate(self, items):
        if len(items) != len(self.weights):
            raise ValueError(
                'Invalid number of items, should be {} but get {}.'.format(
                    len(self.weights), len(items)))
        losses = []
        loss = 0.0
        for item, weight in zip(items, self.weights):
            item_loss = flow.sum(item**2) * weight / item.shape[0]
            loss += item_loss
            losses.append(item_loss)

        return losses, loss
