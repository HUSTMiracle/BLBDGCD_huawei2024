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

from iot_device_sdk_python.gateway.requests.device_info import DeviceInfo


class AddedSubDeviceInfoRsp:
    def __init__(self):
        self._device_info = DeviceInfo()
        self._extension_info: str = ""

    @property
    def parent_device_id(self):
        return self._device_info.parent_device_id

    @parent_device_id.setter
    def parent_device_id(self, value):
        self._device_info.parent_device_id = value

    @property
    def node_id(self):
        return self._device_info.node_id

    @node_id.setter
    def node_id(self, value):
        self._device_info.node_id = value

    @property
    def device_id(self):
        return self._device_info.device_id

    @device_id.setter
    def device_id(self, value):
        self._device_info.device_id = value

    @property
    def name(self):
        return self._device_info.name

    @name.setter
    def name(self, value):
        self._device_info.name = value

    @property
    def description(self):
        return self._device_info.description

    @description.setter
    def description(self, value):
        self._device_info.description = value

    @property
    def manufacturer_id(self):
        return self._device_info.manufacturer_id

    @manufacturer_id.setter
    def manufacturer_id(self, value):
        self._device_info.manufacturer_id = value

    @property
    def model(self):
        return self._device_info.model

    @model.setter
    def model(self, value):
        self._device_info.model = value

    @property
    def product_id(self):
        return self._device_info.product_id

    @product_id.setter
    def product_id(self, value):
        self._device_info.product_id = value

    @property
    def fw_version(self):
        return self._device_info.fw_version

    @fw_version.setter
    def fw_version(self, value):
        self._device_info.fw_version = value

    @property
    def sw_version(self):
        return self._device_info.sw_version

    @sw_version.setter
    def sw_version(self, value):
        self._device_info.sw_version = value

    @property
    def status(self):
        return self._device_info.status

    @status.setter
    def status(self, value):
        self._device_info.status = value

    @property
    def extension_info(self):
        return self._extension_info

    @extension_info.setter
    def extension_info(self, value):
        self._extension_info = value

    def to_dict(self):
        ret = self._device_info.to_dict()
        ret["extension_info"] = self._extension_info
        return ret

    def convert_from_dict(self, json_dict: dict):
        self._device_info.convert_from_dict(json_dict)
        for key in json_dict.keys():
            if key == "extension_info":
                self.extension_info = json_dict.get(key)















