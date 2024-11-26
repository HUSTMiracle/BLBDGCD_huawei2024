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

from iot_device_sdk_python.client.request.command import Command
from typing import Optional


class Action:
    def __init__(self):
        self._type: str = ""
        self._status: str = ""
        self._device_id: str = ""
        self._command: Optional[Command] = None

    @property
    def type(self):
        """
        端侧规则类型
        """
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def status(self):
        """
        端侧规则状态
        """
        return self._status

    @status.setter
    def status(self, value):
        self._status = value

    @property
    def device_id(self):
        """
        设备ID
        """
        return self._device_id

    @device_id.setter
    def device_id(self, value):
        self._device_id = value

    @property
    def command(self):
        """
        命令
        """
        return self._command

    @command.setter
    def command(self, value):
        self._command = value
