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

class ServiceProperty:
    """
    服务属性
    """
    def __init__(self, service_id: str = "", properties: dict = None, event_time: str = None):
        self._service_id: str = service_id
        self._properties: dict = properties
        self._event_time: str = event_time

    @property
    def service_id(self):
        """
        设备的服务ID，在设备关联的产品模型中定义
        """
        return self._service_id

    @service_id.setter
    def service_id(self, value):
        self._service_id = value

    @property
    def properties(self):
        """
        属性值，具体字段由设备模型定义
        """
        return self._properties

    @properties.setter
    def properties(self, value):
        self._properties = value

    @property
    def event_time(self):
        """
        设备采集数据UTC时间（格式为毫秒级别：yyyy-MM-dd'T'HH:mm:ss.SSS'Z'），
        如：20161219T114920Z或者2020-08-12T12:12:12.333Z。

        设备上报数据不带该参数或参数格式错误时，则数据上报时间以平台时间为准。
        """
        return self._event_time

    @event_time.setter
    def event_time(self, value):
        self._event_time = value

    def to_dict(self):
        """
        将请求内容放到字典中

        Returns:
            dict: 字典形式的请求
        """
        return {"service_id": self._service_id, "properties": self._properties, "event_time": self._event_time}

    def convert_from_dict(self, json_dict: dict):
        json_name = ["service_id", "properties", "event_time"]
        for key in json_dict.keys():
            if key not in json_name:
                continue
            if key == "service_id":
                self.service_id = json_dict.get(key)
            elif key == "properties":
                self.properties = json_dict.get(key)
            elif key == "event_time":
                self.event_time = json_dict.get(key)
            else:
                pass

