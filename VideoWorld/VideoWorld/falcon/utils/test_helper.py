# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
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

from mmengine.dist import get_dist_info

from .collect import dist_forward_collect, nondist_forward_collect


def single_gpu_test(model, data_loader):
    model.eval()

    # the function sent to collect function
    def test_mode_func(**x):
        return model(mode='test', **x)

    results = nondist_forward_collect(test_mode_func, data_loader,
                                      len(data_loader.dataset))
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    model.eval()

    # the function sent to collect function
    def test_mode_func(**x):
        return model(mode='test', **x)

    rank, world_size = get_dist_info()
    results = dist_forward_collect(test_mode_func, data_loader, rank,
                                   len(data_loader.dataset))
    return results
