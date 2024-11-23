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

class DeviceEvent:
    """
    服务的事件
    """
    def __init__(self):
        self._service_id: str = ""
        self._event_type: str = ""
        self._event_time: str = ""
        self._event_id: str = ""
        self._paras: dict = dict()

    @property
    def service_id(self):
        """
        事件所属的服务
        """
        return self._service_id

    @service_id.setter
    def service_id(self, value):
        self._service_id = value

    @property
    def event_type(self):
        """
        事件类型
        """
        return self._event_type

    @event_type.setter
    def event_type(self, value):
        self._event_type = value

    @property
    def event_time(self):
        """
        事件发生的时间
        """
        return self._event_time

    @event_time.setter
    def event_time(self, value):
        self._event_time = value

    @property
    def event_id(self):
        """
        事件id，通过该参数关联对应的事件请求
        """
        return self._event_id

    @event_id.setter
    def event_id(self, value):
        self._event_id = value

    @property
    def paras(self):
        """
        事件具体的参数
        """
        return self._paras

    @paras.setter
    def paras(self, value):
        self._paras = value

    def to_dict(self):
        """
        将请求内容放到字典中

        Returns:
            dict: 字典形式的请求
        """
        return {"service_id": self._service_id, "event_type": self._event_type, "event_time": self._event_time,
                "event_id": self._event_id, "paras": self._paras}

    def convert_from_dict(self, json_dict: dict):
        json_name = ["service_id", "event_type", "event_time", "event_id", "paras"]
        for key in json_dict.keys():
            if key not in json_name:
                continue
            if key == "service_id":
                self.service_id = json_dict.get(key)
            elif key == "event_type":
                self.event_type = json_dict.get(key)
            elif key == "event_time":
                self.event_time = json_dict.get(key)
            elif key == "event_id":
                self.event_id = json_dict.get(key)
            elif key == "paras":
                self.paras = json_dict.get(key)
            else:
                pass
