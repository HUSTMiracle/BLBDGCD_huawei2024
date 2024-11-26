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

"""
原始消息类
"""

from paho.mqtt.client import PayloadType


class RawMessage:
    def __init__(self, topic: str, payload: str, qos: int = 1):
        self._topic: str = topic
        self._payload: PayloadType = payload
        self._qos: int = qos

    @property
    def topic(self):
        """
        消息主题
        """
        return self._topic

    @topic.setter
    def topic(self, value):
        self._topic = value

    @property
    def payload(self):
        """
        消息体
        """
        return self._payload

    @payload.setter
    def payload(self, value):
        self._payload = value

    @property
    def qos(self):
        """
        qos
        """
        return self._qos

    @qos.setter
    def qos(self, value):
        self._qos = value
