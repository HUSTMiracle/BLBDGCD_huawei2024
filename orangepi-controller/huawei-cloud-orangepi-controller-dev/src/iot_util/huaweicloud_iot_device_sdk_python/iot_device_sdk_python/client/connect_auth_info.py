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


class ConnectAuthInfo:
    """
    连接鉴权配置
    """
    SECRET_AUTH = 0
    X509_AUTH = 1
    PROTOCOL_MQTT = "MQTT"
    BS_MODE_DIRECT_CONNECT = 0
    BS_MODE_STANDARD_BOOTSTRAP = 1
    BS_MODE_BOOTSTRAP_WITH_SCOPEID = 2

    def __init__(self):
        """ id，在平台注册设备获得 """
        self._id: Optional[str] = None

        """ 认证的类型：0表示密码方式，1表示x509证书方式；默认为0 """
        self._auth_type: int = self.SECRET_AUTH

        """ 设备密码 """
        self._secret: Optional[str] = None

        """ x509证书的pem文件路径 """
        self._cert_path: Optional[str] = None

        """ x509证书的key文件路径 """
        self._key_path: Optional[str] = None

        """ iot平台的ca证书存放路径，用于设备侧校验平台 """
        self._iot_cert_file: Optional[str] = None

        """ 设备自注册场景下使用 """
        self._scope_id: Optional[str] = None

        """ 设备接入平台地址（不包括端口） """
        self._server_uri: Optional[str] = None

        """ 端口 """
        self._port: Optional[int] = None

        """ 协议类型，不填则默认为MQTT """
        self._protocol: str = self.PROTOCOL_MQTT

        """ 0表示直连模式，1表示标准设备发放流程，2表示通过自注册的方式进行批量发放；默认为0 """
        self._bs_mode: int = self.BS_MODE_DIRECT_CONNECT

        """ 设备发放平台的证书路径 """
        self._bs_cert_path: Optional[str] = None

        """ 设备发放时上报的消息 ，在静态策略数据上报方式中，需要在上报的属性 “baseStrategyKeyword” 包含设置的关键字"""
        self._bs_message: Optional[str] = None

        """ 是否校验时间戳，"0"表示HMACSHA256不校验时间戳，"1"表示HMACSHA256校验时间戳；默认为"1"。 """
        self._check_timestamp: str = "1"

        """ 是否支持重连，True表示支持重连， False表示不支持重连"""
        self._reconnect_on_failure = True
        """ 最小退避时间， 默认1s"""
        self._min_backoff = 1 * 1000  # 1s
        """ 最大退避时间，默认30s"""
        self._max_backoff = 30 * 1000
        """ 是否开启端侧规则"""
        self._enable_rule_manage = False
        """ max buffer max"""
        self._max_buffer_message = 0
        """ qos1时最多可以同时发布多条消息，默认20条 """
        self._inflight_messages: Optional[int] = 20
        """ 是否自动上报版本号"""
        self._auto_report_device_info: Optional[bool] = False

    @property
    def id(self):
        """
        id，在平台注册设备获得
        """
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def auth_type(self):
        """
        认证的类型：0表示密码方式，1表示x509证书方式；默认为0
        """
        return self._auth_type

    @auth_type.setter
    def auth_type(self, value):
        self._auth_type = value

    @property
    def secret(self):
        """
        设备密码
        """
        return self._secret

    @secret.setter
    def secret(self, value):
        self._secret = value

    @property
    def cert_path(self):
        """
        x509证书的pem文件路径
        """
        return self._cert_path

    @cert_path.setter
    def cert_path(self, value):
        self._cert_path = value

    @property
    def key_path(self):
        """
        x509证书的key文件路径
        """
        return self._key_path

    @key_path.setter
    def key_path(self, value):
        self._key_path = value

    @property
    def iot_cert_path(self):
        """
        iot平台的ca证书存放路径，用于设备侧校验平台
        """
        return self._iot_cert_file

    @iot_cert_path.setter
    def iot_cert_path(self, value):
        self._iot_cert_file = value

    @property
    def server_uri(self):
        """
        设备接入平台地址（不包括端口）
        """
        return self._server_uri

    @server_uri.setter
    def server_uri(self, value):
        self._server_uri = value

    @property
    def port(self):
        """
        端口
        """
        return self._port

    @port.setter
    def port(self, value):
        self._port = value

    @property
    def protocol(self):
        """
        协议类型，不填则默认为MQTT
        """
        return self._protocol

    @protocol.setter
    def protocol(self, value):
        self._protocol = value

    @property
    def scope_id(self):
        """
        设备自注册场景下使用
        """
        return self._scope_id

    @scope_id.setter
    def scope_id(self, value):
        self._scope_id = value

    @property
    def bs_mode(self):
        """
        是否为设备发放场景，默认为1
        """
        return self._bs_mode

    @bs_mode.setter
    def bs_mode(self, value):
        self._bs_mode = value

    @property
    def bs_cert_path(self):
        """
        设备发放平台的证书路径
        """
        return self._bs_cert_path

    @bs_cert_path.setter
    def bs_cert_path(self, value):
        self._bs_cert_path = value

    @property
    def bs_message(self):
        """
        静态策略数据上报方式下上报的数据
        """
        return self._bs_message

    @bs_message.setter
    def bs_message(self, value):
        self._bs_message = value

    @property
    def check_timestamp(self):
        """
        是否校验时间戳，默认为"1"
        """
        return self._check_timestamp

    @check_timestamp.setter
    def check_timestamp(self, value):
        self._check_timestamp = value

    @property
    def reconnect_on_failure(self):
        """
        是否支持重连，TRUE支持重连
        """
        return self._reconnect_on_failure

    @reconnect_on_failure.setter
    def reconnect_on_failure(self, value):
        self._reconnect_on_failure = value

    @property
    def min_backoff(self):
        """
        最小退避时间
        """
        return self._min_backoff

    @min_backoff.setter
    def min_backoff(self, value):
        self._min_backoff = value

    @property
    def max_backoff(self):
        """
        最大退避时间
        """
        return self._max_backoff

    @max_backoff.setter
    def max_backoff(self, value):
        self._max_backoff = value

    @property
    def enable_rule_manage(self):
        """
        是否支持端侧规则
        """
        return self._enable_rule_manage

    @enable_rule_manage.setter
    def enable_rule_manage(self, value):
        self._enable_rule_manage = value

    @property
    def max_buffer_message(self):
        """
        最大缓存消息，默认为0，不缓存消息
        断链时，生产失败的消息存放队列，待重连后重新发送
        """
        return self._max_buffer_message

    @max_buffer_message.setter
    def max_buffer_message(self, value):
        self._max_buffer_message = value

    @property
    def inflight_messages(self):
        """
        qos1时最多可以同时发布多条消息，默认20条
        """
        return self._inflight_messages

    @inflight_messages.setter
    def inflight_messages(self, value):
        self._inflight_messages = value

    @property
    def auto_report_device_info(self):
        """
        qos1时最多可以同时发布多条消息，默认20条
        """
        return self._auto_report_device_info

    @auto_report_device_info.setter
    def auto_report_device_info(self, value):
        self._auto_report_device_info = value
