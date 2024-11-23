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

from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.client.mqtt_connect_conf import MqttConnectConf


class ClientConf:
    """
    客户端配置
    """

    def __init__(self, connect_auth_info: ConnectAuthInfo, mqtt_connect_conf: Optional[MqttConnectConf] = None):
        self.__connect_auth_info = connect_auth_info
        self.__mqtt_connect_conf = MqttConnectConf()
        if mqtt_connect_conf is not None:
            self.__mqtt_connect_conf = mqtt_connect_conf

    @property
    def connect_auth_info(self):
        return self.__connect_auth_info

    @connect_auth_info.setter
    def connect_auth_info(self, value):
        self.__connect_auth_info = value

    @property
    def mqtt_connect_conf(self):
        return self.__mqtt_connect_conf

    @mqtt_connect_conf.setter
    def mqtt_connect_conf(self, value):
        self.__mqtt_connect_conf = value
