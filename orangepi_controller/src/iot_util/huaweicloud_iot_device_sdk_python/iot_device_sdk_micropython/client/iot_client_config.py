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

from iot_device_sdk_micropython.utils.iot_util import str_is_empty


class IoTClientConfig:
    def __init__(self, server_ip='', device_id='', secret='', port=1883):
        '''
        配置客户端相关信息
        serverIp: IoT平台mqtt对接地址
        device_id: 创建设备时获得的deviceId
        secret: 创建设备时设置的密钥（要替换为自己注册的设备ID与密钥）
        port: mqtt端口，必须为int类型
        '''
        self.__server_ip = server_ip
        self.__device_id = device_id
        self.__secret = secret
        self.__port = port

    @property
    def server_ip(self):
        if self.__server_ip == '':
            raise ValueError('You have not set the server_ip')
        return self.__server_ip

    @server_ip.setter
    def server_ip(self, value):
        if str_is_empty(value):
            raise ValueError('server_ip Wrong !!!, the server_ip is empty')
        self.__server_ip = value

    @property
    def device_id(self):
        if self.__device_id == '':
            raise ValueError('You have not set the device_id')
        return self.__device_id

    @device_id.setter
    def device_id(self, value):
        if str_is_empty(value):
            raise ValueError('device_id Wrong !!!, the device_id is empty')
        self.__device_id = value

    @property
    def secret(self):
        if self.__secret == '':
            raise ValueError('You have not set the secret')
        return self.__secret

    @secret.setter
    def secret(self, value):
        if str_is_empty(value):
            raise ValueError('secret Wrong !!!, the secret is empty')
        self.__secret = value

    @property
    def port(self):
        return self.__port

    @port.setter
    def port(self, value):
        if isinstance(value, int):
            raise ValueError(
                'Invalid port !!!, port should be an instance of type int')
        self.__port = value
