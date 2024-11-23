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

from iot_device_sdk_python.gateway.requests.scan_sub_device_notify import ScanSubDeviceNotify


class SubDevDiscoveryListener(metaclass=ABCMeta):
    """
    暂没有实现
    """
    @abstractmethod
    def on_scan(self, scan_sub_device_notify: ScanSubDeviceNotify):
        """
        平台通知网关扫描子设备

        Args:
            scan_sub_device_notify:  子设备扫描通知

        Returns:
            int: 0表示处理成功，其他表示处理失败
        """
