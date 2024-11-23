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
import logging

from iot_device_sdk_python.timesync.time_sync_listener import TimeSyncListener
from iot_device_sdk_python.service.abstract_service import AbstractService
from iot_device_sdk_python.client.request.device_event import DeviceEvent
from iot_device_sdk_python.transport.action_listener import ActionListener
from iot_device_sdk_python.utils.iot_util import get_event_time, get_gmt_timestamp


class TimeSyncService(AbstractService):
    """
    时间同步服务，提供简单的时间同步服务
    """
    _logger = logging.getLogger(__name__)

    def __init__(self):
        super().__init__()
        self._listener: Optional[TimeSyncListener] = None

    def get_listener(self):
        return self._listener

    def set_listener(self, listener: TimeSyncListener):
        """
        设置时间同步响应监听器

        Args:
            listener:    监听器
        """
        self._listener = listener

    def request_time_sync(self, listener: Optional[ActionListener] = None):
        """
        发起时间同步请求，使用TimeSyncListener接收响应
        Args:
            listener:   发布监听器
        """
        device_event = DeviceEvent()
        device_event.service_id = self.service_id
        device_event.event_type = "time_sync_request"
        device_event.event_time = get_event_time()
        device_send_time = get_gmt_timestamp()
        device_event.paras = {"device_send_time": str(device_send_time)}
        self.get_iot_device().get_client().report_event(device_event, listener)

    def on_event(self, device_event: DeviceEvent):
        """
        时间同步服务的事件回调方法

        Args:
            device_event: 设备事件
        """
        if self._listener is None:
            self._logger.warning("listener in TimeSyncService is None, can not process")
            return
        if not isinstance(self._listener, TimeSyncListener):
            self._logger.warning("listener in TimeSyncService is not TimeSyncListener, can not process")
            return
        if device_event.event_type == "time_sync_response":
            paras: dict = device_event.paras
            device_send_time = int(paras["device_send_time"])
            server_recv_time = int(paras["server_recv_time"])
            server_send_time = int(paras["server_send_time"])
            self._listener.on_time_sync_response(device_send_time, server_recv_time, server_send_time)

