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
from abc import ABCMeta, abstractmethod

from iot_device_sdk_python.gateway.requests.sub_devices_info import SubDevicesInfo
from iot_device_sdk_python.gateway.requests.device_info import DeviceInfo


class SubDevicesPersistence(metaclass=ABCMeta):
    """
    提供子设备信息持久化保存
    """

    @abstractmethod
    def get_sub_device(self, node_id: str) -> DeviceInfo:
        """
        根据设备标识码查询子设备

        Args:
            node_id: 设备标识码

        Returns:
            DeviceInfo: 子设备信息
        """

    @abstractmethod
    def add_sub_devices(self, sub_devices_info: SubDevicesInfo):
        """
        添加子设备处理回调，子类可以重写此接口进行扩展

        Args:
            sub_devices_info:    子设备信息

        Returns:
            int: 处理结果，0表示成功
        """

    @abstractmethod
    def delete_sub_devices(self, sub_devices_info: SubDevicesInfo):
        """
        删除子设备处理回调，子类可以重写此接口进行扩展

        Args:
            sub_devices_info:    子设备信息

        Returns:
            int: 处理结果，0表示成功
        """

    @abstractmethod
    def get_version(self):
        pass
