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


class TimeSyncListener(metaclass=ABCMeta):
    """
    监听事件同步事件
    """
    @abstractmethod
    def on_time_sync_response(self, device_send_time: int, server_recv_time: int, server_send_time: int):
        """
        时间同步响应
        假设设备收到的设备侧时间为device_recv_time，则设备计算自己的准确时间为：
        (server_recv_time + server_send_time + device_recv_time - device_send_time) / 2

        Args:
            device_send_time:    设备发送时间
            server_recv_time:    服务端接收时间
            server_send_time:    服务端响应发送时间
        """
