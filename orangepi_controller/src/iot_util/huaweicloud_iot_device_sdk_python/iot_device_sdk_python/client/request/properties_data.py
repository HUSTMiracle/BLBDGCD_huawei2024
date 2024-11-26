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

class PropertiesData:
    """
    属性数据
    """
    def __init__(self):
        self._properties: dict = dict()
        self._event_time: str = ""

    @property
    def properties(self):
        """
        设备服务的属性列表，具体字段在设备关联的产品模型里定义，可以设置多个字段
        """
        return self._properties

    @properties.setter
    def properties(self, value):
        self._properties = value

    @property
    def event_time(self):
        return self._event_time

    @event_time.setter
    def event_time(self, value):
        self._event_time = value

    def convert_from_dict(self, json_dict: dict):
        json_name = ["properties", "event_time"]
        for key in json_dict.keys():
            if key not in json_name:
                continue
            if key == "properties":
                self.properties = json_dict.get(key)
            elif key == "event_time":
                self.event_time = json_dict.get(key)
            else:
                pass
