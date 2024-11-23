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

"""
网关添加子设备信息
"""


class AddedSubDeviceInfo:
    def __init__(self):
        self._parent_device_id: str = ""
        self._node_id: str = ""
        self._device_id: str = ""
        self._name: str = ""
        self._description: str = ""
        self._product_id: str = ""
        self._extension_info: str = ""

    @property
    def parent_device_id(self):
        return self._parent_device_id

    @parent_device_id.setter
    def parent_device_id(self, value):
        self._parent_device_id = value

    @property
    def node_id(self):
        return self._node_id

    @node_id.setter
    def node_id(self, value):
        self._node_id = value

    @property
    def device_id(self):
        return self._device_id

    @device_id.setter
    def device_id(self, value):
        self._device_id = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def description(self):
        return self.description

    @description.setter
    def description(self, value):
        self._description = value

    @property
    def product_id(self):
        return self._product_id

    @product_id.setter
    def product_id(self, value):
        self._product_id = value

    @property
    def extension_info(self):
        return self._extension_info

    @extension_info.setter
    def extension_info(self, value):
        self._extension_info = value

    def to_dict(self):
        return {"parent_device_id": self._parent_device_id,
                "node_id": self._node_id,
                "device_id": self._device_id,
                "name": self._name,
                "description": self._description,
                "product_id": self._product_id,
                "extension_info": self._extension_info}

    def convert_from_dict(self, json_dict: dict):
        json_name = ["parent_device_id", "node_id", "device_id", "name",
                     "description", "product_id", "extension_info"]
        for key in json_dict.keys():
            if key not in json_name:
                continue
            if key == "parent_device_id":
                self.parent_device_id = json_dict.get(key)
            elif key == "node_id":
                self.node_id = json_dict.get(key)
            elif key == "device_id":
                self.device_id = json_dict.get(key)
            elif key == "name":
                self.name = json_dict.get(key)
            elif key == "description":
                self.description = json_dict.get(key)
            elif key == "product_id":
                self.product_id = json_dict.get(key)
            elif key == "extension_info":
                self.extension_info = json_dict.get(key)
            else:
                pass
