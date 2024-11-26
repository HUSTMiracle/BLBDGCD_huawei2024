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


class MqttConnectConf:
    """
    mqtt 配置
    """

    def __init__(self):
        """ 保活时间，仅MQTT协议，sdk默认填写120（单位：秒）。可选30~1200（秒）范围 """
        self._keep_alive_time: int = 120

        """ 客户端qos，0或1，默认为1 """
        self._qos: int = 1

        """ 连接超时时间 """
        self._timeout: float = 1.0

    @property
    def keep_alive_time(self):
        """ 保活时间，仅MQTT协议，sdk默认填写120（单位：秒）。可选30~1200（秒）范围 """
        return self._keep_alive_time

    @keep_alive_time.setter
    def keep_alive_time(self, value):
        self._keep_alive_time = value

    @property
    def qos(self):
        """ 客户端qos，0或1，默认为1 """
        return self._qos

    @qos.setter
    def qos(self, value):
        self._qos = value

    @property
    def timeout(self):
        """ 连接超时时间 """
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self._timeout = value
