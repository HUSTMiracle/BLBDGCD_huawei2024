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

from abc import abstractmethod, ABCMeta

from iot_device_sdk_python.client.request.raw_device_message import RawDeviceMessage


class RawDeviceMessageListener(metaclass=ABCMeta):
    """
    设备消息监听器，用于接收平台下发的设备消息
    """

    @abstractmethod
    def on_raw_device_message(self, message: RawDeviceMessage):
        """
        处理平台下发的设备消息

        Args:
            message:     设备消息内容
        """
