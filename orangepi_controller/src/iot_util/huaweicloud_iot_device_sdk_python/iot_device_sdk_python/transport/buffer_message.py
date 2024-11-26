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

from iot_device_sdk_python.transport.raw_message import RawMessage
from iot_device_sdk_python.transport.action_listener import ActionListener


class BufferMessage:
    def __init__(self, raw_message: RawMessage, listener: ActionListener):
        self._raw_message = raw_message
        self._listener = listener

    @property
    def raw_message(self):
        """
        消息主题
        """
        return self._raw_message

    @raw_message.setter
    def raw_message(self, value):
        self._raw_message = value

    @property
    def listener(self):
        """
        消息主题
        """
        return self._listener

    @listener.setter
    def listener(self, value):
        self._listener = value
