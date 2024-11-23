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
from typing import List

from iot_device_sdk_python.gateway.requests.device_info import DeviceInfo


class SubDevicesInfo:
    """
    子设备信息
    """
    def __init__(self):
        self._devices: List[DeviceInfo] = []
        self._version: int = 0

    @property
    def devices(self):
        return self._devices

    @devices.setter
    def devices(self, value):
        self._devices = value

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, value):
        self._version = value















