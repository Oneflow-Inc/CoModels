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
from core.utils import ones
from core.nn import FC, WeightedL2
from core.pinns import Interval, Variables, UndeterminedLorenzSystem, DirichletBC, Supervision, PINNSolver

# define geometry
domain = Interval(0, 3).discrete(interior=500, boundary=2)

# define pde
var = Variables(name_value_map={'C1': 1.0, 'C2': 1.0, 'C3': 1.0})
pde = UndeterminedLorenzSystem(var)


# set bc
def bc_constrain_func(x):
    return np.isclose(x[:, 0], np.zeros(x.shape[0]))


def bc_value_func(x):
    v1 = ones(x.shape[0]) * (-8.0)
    v2 = ones(x.shape[0]) * (7.0)
    v3 = ones(x.shape[0]) * (27.0)
    return flow.stack((v1, v2, v3), dim=1)


bc = DirichletBC(domain, bc_constrain_func, bc_value_func)

# define sup
data = np.load("data/lorenz.npz")
sup = Supervision(data['t'], data['y'])

# define nn
net = FC(num_ins=1,
         num_outs=3,
         num_layers=5,
         hiden_size=40,
         activiation=flow.nn.Tanh())

# define loss
loss = WeightedL2(weights=[1, 1, 1, 1, 100])

# define optimizer
opt = flow.optim.Adam(
    list(net.parameters()) + list(pde.get_variables().parameters()), 0.001)

# define solver
solver = PINNSolver(network=net,
                    loss=loss,
                    optimizer=opt,
                    domain=domain,
                    pdes=pde,
                    inputs_names=['t'],
                    outputs_names=['x', 'y', 'z'],
                    bcs=[bc],
                    bc_indexes=[
                        [0, 1, 2],
                    ],
                    sups=[sup])

# train
solver.train(num_epoch=20000, log_frequency=100, checkpoint_frequency=1000)

# infer
# solver.load_checkpoint('log/checkpoint_10000.pt')
solver.evaluate()
