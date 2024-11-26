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

from iot_device_sdk_python.client.request.properties_data import PropertiesData


class ShadowData:
    """
    影子数据
    """
    def __init__(self):
        self._service_id: str = ""
        self._desired: Optional[PropertiesData] = None
        self._reported: Optional[PropertiesData] = None
        self._version: Optional[int] = None

    @property
    def service_id(self):
        """
        服务id
        """
        return self._service_id

    @service_id.setter
    def service_id(self, value):
        self._service_id = value

    @property
    def desired(self):
        """
        设备影子desired区的属性列表
        """
        return self._desired

    @desired.setter
    def desired(self, value):
        self._desired = value

    @property
    def reported(self):
        """
        设备影子reported区的属性列表
        """
        return self._reported

    @reported.setter
    def reported(self, value):
        self._reported = value

    @property
    def version(self):
        """
        设备影子版本信息
        """
        return self._version

    @version.setter
    def version(self, value):
        self._version = value

    def convert_from_dict(self, json_dict: dict):
        json_name = ["service_id", "desired", "reported", "version"]
        for key in json_dict.keys():
            if key not in json_name:
                continue
            if key == "service_id":
                self.service_id = json_dict.get(key)
            elif key == "desired":
                desired = PropertiesData()
                desired.convert_from_dict(json_dict.get(key))
                self.desired = desired
            elif key == "reported":
                reported = PropertiesData()
                reported.convert_from_dict(json_dict.get(key))
                self.reported = reported
            elif key == "version":
                self.version = json_dict.get(key)
            else:
                pass
