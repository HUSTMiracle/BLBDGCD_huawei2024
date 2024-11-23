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

from iot_device_sdk_python.client.request.service_property import ServiceProperty


class DeviceProperty:
    """
    设备属性
    """
    def __init__(self):
        self._device_id: str = ""
        self._services: List[ServiceProperty] = []

    @property
    def device_id(self):
        return self._device_id

    @device_id.setter
    def device_id(self, value):
        self._device_id = value

    @property
    def services(self):
        return self._services

    @services.setter
    def services(self, value):
        self._services = value

    def to_dict(self):
        service_list = list()
        for service in self._services:
            service_list.append(service.to_dict())
        return {"device_id": self._device_id, "services": service_list}


