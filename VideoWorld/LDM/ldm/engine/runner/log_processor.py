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

from mmengine.runner import LogProcessor as BaseLogProcessor

from ldm.registry import LOG_PROCESSORS


@LOG_PROCESSORS.register_module()  # type: ignore
class LogProcessor(BaseLogProcessor):
    """LogProcessor inherits from :class:`mmengine.runner.LogProcessor` and
    overwrites :meth:`self.get_log_after_iter`.

    This log processor should be used along with
    :class:`mmagic.engine.runner.MultiValLoop` and
    :class:`mmagic.engine.runner.MultiTestLoop`.
    """

    def _get_dataloader_size(self, runner, mode) -> int:
        """Get dataloader size of current loop. In `MultiValLoop` and
        `MultiTestLoop`, we use `total_length` instead of `len(dataloader)` to
        denote the total number of iterations.

        Args:
            runner (Runner): The runner of the training/validation/testing
            mode (str): Current mode of runner.

        Returns:
            int: The dataloader size of current loop.
        """
        if hasattr(self._get_cur_loop(runner, mode), 'total_length'):
            return self._get_cur_loop(runner, mode).total_length
        else:
            return super()._get_dataloader_size(runner, mode)
