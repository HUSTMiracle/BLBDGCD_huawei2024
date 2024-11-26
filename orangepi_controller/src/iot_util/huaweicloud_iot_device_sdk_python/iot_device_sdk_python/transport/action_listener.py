# -*- encoding: utf-8 -*-

# Copyright (c) 2023-2024 Huawei Cloud Computing Technology Co., Ltd. All rights reserved.
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

from __future__ import absolute_import
from typing import Optional
from abc import abstractmethod, ABCMeta


class ActionListener(metaclass=ABCMeta):
    """
    动作监听器，用户接收动作执行结果
    """
    @abstractmethod
    def on_success(self, message: str):
        """
        执行成功通知
        :param message: 上下文信息
        """

    @abstractmethod
    def on_failure(self, message: str, e: Optional[Exception]):
        """
        执行失败通知
        :param message: 上下文信息
        :param e:       Exception
        """
