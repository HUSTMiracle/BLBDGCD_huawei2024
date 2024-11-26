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
from abc import abstractmethod, ABCMeta
from typing import List

from iot_device_sdk_python.client.request.command import Command
from iot_device_sdk_python.client.request.device_event import DeviceEvent


class IService(metaclass=ABCMeta):
    """
    服务接口类
    """

    @abstractmethod
    def on_read(self, properties: List[str]):
        """
        读属性回调

        Args:
            properties:  指定读取的属性名，不指定则读取全部可读属性
        Returns:
            dict:    属性值，字典的形式
        """

    @abstractmethod
    def on_write(self, properties: dict):
        """
        写属性回调

        Args:
            properties:  属性期望值
        Returns:
            IotResult:  执行结果
        """

    @abstractmethod
    def on_command(self, command: Command):
        """
        命令回调

        Args:
            command: 命令
        Returns:
            CommandRsp: 执行结果
        """

    @abstractmethod
    def on_event(self, device_event: DeviceEvent):
        """
        事件回调

        Args:
            device_event:    事件
        """

