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

class DeviceMessage:
    """
    设备消息
    """
    def __init__(self):
        self._object_device_id: str = ""
        self._id: str = ""
        self._name: str = ""
        self._content: str = ""

    @property
    def device_id(self):
        """
        消息对应的最终目标设备，没有携带则表示目标设备即topic中指定的设备
        """
        return self._object_device_id

    @device_id.setter
    def device_id(self, value):
        self._object_device_id = value

    @property
    def id(self):
        """
        消息id，消息的唯一标识
        """
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def name(self):
        """
        消息名称
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def content(self):
        """
        消息内容
        """
        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    def to_dict(self):
        """
        将请求内容放到字典中

        Returns:
            dict: 字典形式的请求
        """
        return {"object_device_id": self._object_device_id, "id": self._id, "name": self._name,
                "content": self._content}

    def convert_from_dict(self, json_dict: dict):
        json_name = ["object_device_id", "name", "id", "content"]
        for key in json_dict.keys():
            if key not in json_name:
                continue
            if key == "object_device_id":
                self.device_id = json_dict.get(key)
            elif key == "id":
                self.id = json_dict.get(key)
            elif key == "name":
                self.name = json_dict.get(key)
            elif key == "content":
                self.content = json_dict.get(key)
            else:
                pass
