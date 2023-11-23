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

import os
import itertools
import numpy as np
import oneflow as flow
from functools import wraps


class Config(object):

    def __init__(self):
        super(Config, self).__init__()
        self.allow_dtypes = ['float32', 'float64']
        self.dtype = 'float32'
        self.dtype_n = np.float32
        self.dtype_t = flow.float32
        self.device = flow.device(
            'cuda:0' if flow.cuda.is_available() else 'cpu')

        self.seed = None
        flow.set_default_tensor_type(flow.FloatTensor)

    def get_dtype(self):
        return self.dtype

    def get_dtype_n(self):
        return self.dtype_n

    def get_dtype_t(self):
        return self.dtype_t

    def set_dtype(self, dtype):
        if dtype not in self.allow_dtypes:
            raise ValueError(
                'Invalid default dtype, should be float32 or float64.')
        self.dtype = dtype
        self.dtype_n = np.float32 if self.dtype == 'float32' else np.float64
        self.dtype_t = flow.float32 if self.dtype == 'float32' else flow.float64
        flow.set_default_tensor_type(flow.FloatTensor if self.dtype ==
                                     'float32' else flow.DoubleTensor)

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)
        flow.manual_seed(seed)

    def set_device(self, device):
        self.device = flow.device(device)

    def get_device(self):
        return self.device


cfg = Config()


def sample(num_samples, dim, method, dtype):
    if method is not 'uniform':
        raise ValueError('Invalid method, only support uniform.')

    # do not include start point 0, and end point
    n = int(np.ceil(num_samples**(1 / dim)))
    sample_list = [
        np.linspace(0.0, 1.0, num=n + 1, endpoint=False, dtype=dtype)[1:]
        for _ in range(dim)
    ]

    ret = list(itertools.product(*sample_list))
    return np.array(ret, dtype=dtype).reshape(-1, dim)


def cache_func(func):
    cache = {}

    @wraps(func)
    def wrapper_cache(*args):
        # id(args) will have error, will be the same
        # generator for will have bug, will not be the same
        key = ' '.join([str(id(arg)) for arg in args])
        if key not in cache:
            cache[key] = func(*args)
        return cache[key]

    return wrapper_cache


def check_indexes(indexes):
    if isinstance(indexes, tuple) or isinstance(indexes, list):
        for index in indexes:
            if not isinstance(index, int):
                return False
    else:
        return False
    return True


def tensor(data,
           dtype=None,
           device=None,
           requires_grad=False,
           pin_memory=False):

    if dtype is None:
        dtype = cfg.get_dtype_t()
    if device is None:
        device = cfg.get_device()

    data = flow.tensor(data,
                       dtype=dtype,
                       device=device,
                       requires_grad=requires_grad,
                       pin_memory=pin_memory)
    return data


def ones(*size, dtype=None, device=None, requires_grad=False):

    if dtype is None:
        dtype = cfg.get_dtype_t()
    if device is None:
        device = cfg.get_device()

    data = flow.ones(*size,
                     dtype=dtype,
                     device=device,
                     requires_grad=requires_grad)
    return data


def zeros(*size, dtype=None, device=None, requires_grad=False):

    if dtype is None:
        dtype = cfg.get_dtype_t()
    if device is None:
        device = cfg.get_device()

    data = flow.zeros(*size,
                      dtype=dtype,
                      device=device,
                      requires_grad=requires_grad)
    return data


def save_checkpoint(path, net, opt, var=None):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if var is None:
        flow.save(
            {
                'net_state_dict': net.state_dict(),
                'opt_state_dict': opt.state_dict()
            }, path)
    else:
        flow.save(
            {
                'net_state_dict': net.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'var_state_dict': var.state_dict()
            }, path)


def load_checkpoint(path, net, opt, var=None):
    checkpoint = flow.load(path)
    net.load_state_dict(checkpoint['net_state_dict'])
    opt.load_state_dict(checkpoint['opt_state_dict'])
    if var is not None:
        var.load_state_dict(checkpoint['var_state_dict'])
