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


from .model_utils import (build_module, default_init_weights,
                          generation_init_weights, get_module_device,
                          get_valid_noise_size, get_valid_num_batches,
                          make_layer, remove_tomesd, set_requires_grad,
                          set_tomesd, set_xformers, xformers_is_enable)
from .sampling_utils import label_sample_fn, noise_sample_fn
from .tensor_utils import get_unknown_tensor, normalize_vecs

__all__ = [
    'default_init_weights', 'make_layer',
    'generation_init_weights', 'set_requires_grad',  'get_unknown_tensor', 'noise_sample_fn',
    'label_sample_fn', 'get_valid_num_batches', 'get_valid_noise_size',
    'get_module_device', 'normalize_vecs', 'build_module', 'set_xformers',
    'xformers_is_enable', 'set_tomesd', 'remove_tomesd'
]
