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
from typing import Optional

from iot_device_sdk_python.transport.connect_action_listener import ConnectActionListener
from iot_device_sdk_python.devicelog.device_log_service import DeviceLogService
from iot_device_sdk_python.utils.iot_util import get_gmt_timestamp


class DefaultConnActionLogListener(ConnectActionListener):
    def __init__(self, device_log_service: DeviceLogService):
        self._device_log_service = device_log_service

    def on_success(self, token: int):
        """
        首次建链成功

        Args:
            token:   返回token
        """
        # 只有当connect_failed_dict不为空时report一次设备日志，上报建链失败的原因
        # 若建链成功，这里不上报任何设备日志。因为会与DefaultConnLogListener的connect_complete()上报重复的日志，造成浪费
        if self._device_log_service.connect_failed_dict is not None:
            tmp = list(self._device_log_service.connect_failed_dict.keys())
            timestamp = tmp[0]
            self._device_log_service.report_device_log(timestamp, "DEVICE_STATUS",
                                                       self._device_log_service.connect_failed_dict.get(timestamp))

    def on_failure(self, token: int, err: Optional[Exception]):
        """
        首次建链失败

        Args:
            token:   返回token
            err:     失败异常
        """
        failed_dict = dict()
        failed_dict[str(get_gmt_timestamp())] = "connect failed, the reason is " + str(err)
        self._device_log_service.connect_failed_dict = failed_dict











