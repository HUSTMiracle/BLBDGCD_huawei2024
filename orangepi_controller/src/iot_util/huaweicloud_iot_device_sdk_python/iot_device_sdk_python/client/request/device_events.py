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

from iot_device_sdk_python.client.request.device_event import DeviceEvent


class DeviceEvents:
    """
    设备事件
    """
    def __init__(self):
        self._device_id: str = ""
        self._services: List[DeviceEvent] = []

    @property
    def device_id(self):
        """
        事件对应的最终目标设备，没有携带则表示目标设备即topic中指定的设备
        """
        return self._device_id

    @device_id.setter
    def device_id(self, value: str):
        self._device_id = value

    @property
    def services(self):
        """
        事件服务列表
        """
        return self._services

    @services.setter
    def services(self, value):
        self._services = value

    def to_dict(self):
        """
        将请求内容放到字典中

        Returns:
            dict: 字典形式的请求
        """
        service_list = list()
        for service in self._services:
            service_list.append(service.to_dict())
        return {"object_device_id": self._device_id, "services": service_list}

    def convert_from_dict(self, json_dict: dict):
        json_name = ["object_device_id", "services"]
        for key in json_dict.keys():
            if key not in json_name:
                continue
            if key == "object_device_id":
                self.device_id = json_dict.get(key)
            elif key == "services":
                device_event_dict_list = json_dict.get(key)
                for device_event_dict in device_event_dict_list:
                    device_event = DeviceEvent()
                    device_event.convert_from_dict(device_event_dict)
                    self._services.append(device_event)
            else:
                pass
