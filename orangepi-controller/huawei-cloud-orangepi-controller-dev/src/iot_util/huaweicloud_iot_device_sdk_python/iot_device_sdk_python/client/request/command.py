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

class Command:
    """
    设备命令
    """
    def __init__(self):
        self._service_id: str = ""
        self._command_name: str = ""
        self._device_id: str = ""
        self._paras: dict = dict()

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
    def command_name(self):
        """
        设备命令名称，在设备关联的产品模型中定义
        """
        return self._command_name

    @command_name.setter
    def command_name(self, value):
        self._command_name = value

    @property
    def device_id(self):
        """
        命令对应的目标设备ID，命令下发对应的最终目标设备，没有携带则表示目标设备即topic中指定的设备
        """
        return self._device_id

    @device_id.setter
    def device_id(self, value):
        self._device_id = value

    @property
    def paras(self):
        """
        设备命令的执行参数，具体字段在设备关联的产品模型中定义
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
        return {"service_id": self._service_id,
                "command_name": self._command_name,
                "object_device_id": self._device_id,
                "paras": self._paras}

    def convert_from_dict(self, json_dict: dict):
        json_name = ["service_id", "command_name", "object_device_id", "paras"]
        for key in json_dict.keys():
            if key not in json_name:
                continue
            if key == "service_id":
                self.service_id = json_dict.get(key)
            elif key == "command_name":
                self.command_name = json_dict.get(key)
            elif key == "object_device_id":
                self.device_id = json_dict.get(key)
            elif key == "paras":
                self.paras = json_dict.get(key)
            else:
                pass
