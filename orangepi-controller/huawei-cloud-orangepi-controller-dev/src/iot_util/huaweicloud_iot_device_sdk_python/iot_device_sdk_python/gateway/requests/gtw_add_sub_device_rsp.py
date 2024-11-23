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
from typing import List

from iot_device_sdk_python.gateway.requests.added_sub_device_info_rsp import AddedSubDeviceInfoRsp
from iot_device_sdk_python.gateway.requests.add_sub_device_failed_reason import AddSubDeviceFailedReason


class GtwAddSubDeviceRsp:
    def __init__(self):
        self._successful_devices: List[AddedSubDeviceInfoRsp] = []
        self._add_sub_device_failed_reasons: List[AddSubDeviceFailedReason] = []

    @property
    def successful_devices(self):
        return self._successful_devices

    @successful_devices.setter
    def successful_devices(self, value):
        self._successful_devices = value

    @property
    def add_sub_device_failed_reasons(self):
        return self._add_sub_device_failed_reasons

    @add_sub_device_failed_reasons.setter
    def add_sub_device_failed_reasons(self, value):
        self._add_sub_device_failed_reasons = value

    def to_dict(self):
        return {"successful_devices": self._successful_devices,
                "failed_devices": self._add_sub_device_failed_reasons}













