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


import json
import logging
from json import JSONDecodeError

from iot_device_sdk_python.client.request.device_message import DeviceMessage


class RawDeviceMessage:
    _logger = logging.getLogger(__name__)

    __SYSTEM_MESSAGE_KEYS = {"name", "id", "content", "object_device_id"}

    """
    设备消息
    """

    def __init__(self, payload: bytes):
        self._payload = payload

    @property
    def payload(self):
        """
        message下发的原始数据
        """
        return self._payload

    @payload.setter
    def payload(self, payload: bytes):
        self._payload = payload

    def to_utf8_string(self):
        """"
        尝试将原始消息以utf-8格式decode，如无法decode，则raise UnicodeDecodeError
        """
        return self._payload.decode('utf-8')

    def to_device_message(self):
        try:
            device_msg_dict = json.loads(self.to_utf8_string())
        except (JSONDecodeError, UnicodeDecodeError):
            self._logger.debug("device message is not in system format")
            return None  # can't convert the system format

        if any(map(lambda a: a not in self.__SYSTEM_MESSAGE_KEYS, device_msg_dict.keys())):
            self._logger.debug("device message is not in system format because contain unexpected keys")
            return None

        if any(map(lambda a: a is not None and not isinstance(a, str), device_msg_dict.values())):
            self._logger.debug("device message is not in system format because some values are not str")
            return None

        device_msg = DeviceMessage()
        device_msg.convert_from_dict(device_msg_dict)
        return device_msg
