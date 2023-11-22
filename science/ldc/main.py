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

import numpy as np
import oneflow as flow
from core.utils import ones, zeros
from core.nn import FC, WeightedL2
from core.pinns import Rectangle, NavierStokes2D, DirichletBC, PINNSolver
import argparse
import wget
import os

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--type", type=str, default="train")
parser.add_argument("--pretrained", type=bool, default=False)
args = parser.parse_args()

# set bc
def bc_constrain_func_1(x):
    return np.isclose(x[:, 1], np.ones(x.shape[0]) * (-0.05 + 0.1))


def bc_value_func_1(x):
    u = ones(x.shape[0])
    v = zeros(x.shape[0])
    return flow.stack((u, v), dim=1)


def bc_constrain_func_2(x):
    return np.logical_not(
        np.isclose(x[:, 1],
                   np.ones(x.shape[0]) * (-0.05 + 0.1)))


def build_model():
    # define geometry
    domain = Rectangle(origins=[-0.05, -0.05],
                    extents=[0.1, 0.1]).discrete(interior=8000, boundary=400)

    # define pde
    pde = NavierStokes2D(nu=0.01, rho=1.0)

    # define nn
    net = FC(num_ins=2,
            num_outs=3,
            num_layers=5,
            hiden_size=20,
            activiation=flow.nn.Tanh())

    # define loss
    loss = WeightedL2(weights=[1, 1, 1, 100, 100])

    # define optimizer
    opt = flow.optim.Adam(net.parameters(), 0.001)

    bc1 = DirichletBC(domain, bc_constrain_func_1, bc_value_func_1)

    bc2 = DirichletBC(domain, bc_constrain_func_2, lambda x: zeros((x.shape[0], 2)))

    # define solver
    solver = PINNSolver(network=net,
                        loss=loss,
                        optimizer=opt,
                        domain=domain,
                        pdes=pde,
                        inputs_names=['x', 'y'],
                        outputs_names=['u', 'v', 'p'],
                        bcs=[bc1, bc2],
                        bc_indexes=[[0, 1], [0, 1]])
    return solver

if __name__ == "__main__":
    solver = build_model()
    # load pretrained
    if args.pretrained:
        print("Load checkpoint")
        if not os.path.isfile('ldc.of'):
            url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/ldc.of"
            wget.download(url,'ldc.of')
        solver.load_checkpoint('ldc.of')

    if args.type=="train":
        # train
        print("Start train")
        solver.train(num_epoch=30000, log_frequency=100, checkpoint_frequency=1000)
    elif args.type=="infer":
        # infer
        print("Start infer")
        solver.evaluate()
        solver.visualize()

