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

import logging
import json
import os

from iot_device_sdk_python.gateway.sub_devices_persistence import SubDevicesPersistence
from iot_device_sdk_python.gateway.requests.sub_devices_info import SubDevicesInfo
from iot_device_sdk_python.gateway.requests.device_info import DeviceInfo
from iot_gateway_demo.sub_dev_info import SubDevInfo


class SubDevicesFilePersistence(SubDevicesPersistence):
    """
    将子设备信息保存到json文件。用户可以自己实现SubDevicesPersistence接口来进行替换
    """
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._sub_dev_info_cache = SubDevInfo()
        self._conf_file = "./subdevices.json"
        if not os.path.exists(self._conf_file):
            init_sub_dev_info = SubDevInfo()
            init_sub_dev_info.version = 0
            init_sub_dev_info.sub_devices = {"1": DeviceInfo().to_dict()}
            with open(self._conf_file, 'w') as init_f:
                json.dump(init_sub_dev_info.to_dict(), init_f)
        self._read_conf_file()

    def _read_conf_file(self):
        try:
            with open(self._conf_file, 'r') as load_f:
                load_dict = json.load(load_f)
        except Exception as e:
            self._logger.error(e)
            return
        self._sub_dev_info_cache.version = load_dict["version"]
        self._sub_dev_info_cache.sub_devices = load_dict["sub_devices"]

    def _write_conf_file(self):
        try:
            with open(self._conf_file, 'w') as write_f:
                json.dump(self._sub_dev_info_cache.to_dict(), write_f)
                self._logger.info("write file success")
        except Exception as e:
            self._logger.error(e)

    def get_sub_device(self, node_id: str):
        if node_id in self._sub_dev_info_cache.sub_devices.keys():
            device: dict = self._sub_dev_info_cache.sub_devices.get(node_id)
            device_info = DeviceInfo()
            device_info.convert_from_dict(device)
            return device_info
        else:
            return None

    def add_sub_devices(self, sub_devices_info: SubDevicesInfo):
        if sub_devices_info is None:
            return -1

        if 0 < sub_devices_info.version <= self._sub_dev_info_cache.version:
            self._logger.info("version too low: " + str(sub_devices_info.version))
            return -1

        if self.add_sub_device_to_file(sub_devices_info) != 0:
            self._logger.info("write file fail")
            return -1
        return 0

    def delete_sub_devices(self, sub_devices_info: SubDevicesInfo):
        if sub_devices_info is None:
            return -1

        if 0 < sub_devices_info.version <= self._sub_dev_info_cache.version:
            self._logger.info("version too low: " + str(sub_devices_info.version))
            return -1

        if self.rmv_sub_device_to_file(sub_devices_info) != 0:
            self._logger.info("remove from file fail")
            return -1
        return 0

    def get_version(self):
        return self._sub_dev_info_cache.version

    def add_sub_device_to_file(self, sub_devices_info: SubDevicesInfo):
        self._read_conf_file()
        try:
            for device in sub_devices_info.devices:
                device_node_id = device.node_id
                self._sub_dev_info_cache.sub_devices[device_node_id] = device.to_dict()
                self._logger.info("add subdev: " + device_node_id)
            self._sub_dev_info_cache.version = sub_devices_info.version
            self._logger.info("version update to " + str(self._sub_dev_info_cache.version))
        except Exception as e:
            self._logger.error(e)
            return -1
        self._write_conf_file()
        return 0

    def rmv_sub_device_to_file(self, sub_devices_info: SubDevicesInfo):
        self._read_conf_file()
        try:
            for device in sub_devices_info.devices:
                device_node_id = device.node_id
                if device_node_id not in self._sub_dev_info_cache.sub_devices.keys():
                    # 若pop一个不存在的对象会报错
                    self._logger.info("node_id not found in SubDevInfo cache: " + device_node_id)
                else:
                    self._sub_dev_info_cache.sub_devices.pop(device_node_id)
                    self._logger.info("rmv subdev: " + device_node_id)
            self._sub_dev_info_cache.version = sub_devices_info.version
            self._logger.info("version update to " + str(self._sub_dev_info_cache.version))
        except Exception as e:
            self._logger.error(e)
            return -1
        self._write_conf_file()
        return 0
