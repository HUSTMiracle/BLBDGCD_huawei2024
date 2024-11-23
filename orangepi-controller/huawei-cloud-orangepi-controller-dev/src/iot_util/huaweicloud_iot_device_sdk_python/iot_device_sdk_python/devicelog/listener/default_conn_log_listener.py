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
from iot_device_sdk_python.transport.connect_listener import ConnectListener
from iot_device_sdk_python.devicelog.device_log_service import DeviceLogService
from iot_device_sdk_python.utils.iot_util import get_gmt_timestamp


class DefaultConnLogListener(ConnectListener):
    def __init__(self, device_log_service: DeviceLogService):
        self._device_log_service = device_log_service

    def connection_lost(self, cause: str):
        """
        连接丢失通知

        Args:
            cause:   连接丢失原因
        """
        lost_dict = dict()
        str_current_time_millis = str(get_gmt_timestamp())
        lost_dict[str_current_time_millis] = "connect lost"
        self._device_log_service.connect_lost_dict = lost_dict

    def connect_complete(self, reconnect: bool, server_uri: str):
        """
        连接成功通知，如果是断链重连的情景，重连成功会上报断链的时间戳

        Args:
            reconnect:   是否为重连（当前此参数没有作用）
            server_uri:  服务端地址
        """
        self._device_log_service.report_device_log(str(get_gmt_timestamp()), "DEVICE_STATUS",
                                                   "connect complete, the uri is " + str(server_uri))
        if self._device_log_service.connect_lost_dict is not None:
            key_list = list(self._device_log_service.connect_lost_dict.keys())
            timestamp = key_list[0]
            self._device_log_service.report_device_log(timestamp, "DEVICE_STATUS",
                                                       self._device_log_service.connect_lost_dict.get(timestamp))















