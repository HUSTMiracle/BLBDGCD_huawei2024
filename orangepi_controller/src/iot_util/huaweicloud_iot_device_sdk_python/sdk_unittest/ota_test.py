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
from unittest import TestCase, mock

from paho.mqtt.client import MQTTMessage

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.iot_device import IotDevice
from iot_device_sdk_python.ota.ota_listener import OTAListener
from iot_device_sdk_python.ota.ota_package_info import OTAPackageInfo
from iot_device_sdk_python.ota.ota_query_version import OTAQueryVersion
from iot_device_sdk_python.ota.ota_service import OTAService


class OTATestListener(OTAListener):
    """
    一个实现OTA监听器的例子
    """

    def __init__(self, result_dict: dict):
        self.result_dict = result_dict

    def on_query_version(self, queryInfo: OTAQueryVersion):
        """
        接收查询版本通知
        """

    def on_receive_package_info(self, pkg: OTAPackageInfo):
        """
        接收新版本通知
        :param pkg:     新版本包信息
        """
        self.result_dict["url"] = pkg.url


class TestOTAMethod(TestCase):

    @mock.patch("paho.mqtt.client.Client.is_connected")
    @mock.patch("paho.mqtt.client.Client.loop_forever")
    @mock.patch("paho.mqtt.client.Client.connect")
    def setUp(self, connect, loop_forever, is_connected):
        # mock
        connect.return_value = 0
        loop_forever.return_value = 0
        is_connected.return_value = True

        # setup
        server_uri = "access address"
        port = 8883
        self.device_id = "productId_nodeId"
        secret = "12345678"
        iot_ca_cert_path = "./resources/root.pem"

        connect_auth_info = ConnectAuthInfo()
        connect_auth_info.server_uri = server_uri
        connect_auth_info.port = port
        connect_auth_info.id = self.device_id
        connect_auth_info.secret = secret
        connect_auth_info.iot_cert_path = iot_ca_cert_path
        connect_auth_info.auth_type = ConnectAuthInfo.SECRET_AUTH
        connect_auth_info.bs_mode = ConnectAuthInfo.BS_MODE_DIRECT_CONNECT

        client_conf = ClientConf(connect_auth_info)

        self.result_dict = dict()

        self.device = IotDevice(client_conf)
        ota_service: OTAService = self.device.get_ota_service()
        ota_service_listener = OTATestListener(self.result_dict)
        ota_service.set_ota_listener(ota_service_listener)

        self.device.connect()
        return

    def test_receive_ota_package_info(self):
        # test
        _get_connection = getattr(self.device.get_client(), "_get_connection")
        _connection = _get_connection()
        _get_paho_client = getattr(_connection, "_get_paho_client")
        _paho_client = _get_paho_client()
        _handle_on_message = getattr(_paho_client, "_handle_on_message")
        message = MQTTMessage()
        message.topic = b'$oc/devices/' \
                        + self.device_id.encode(encoding="utf-8") \
                        + b'/sys/events/down'
        message.payload = b'{"services":[{' \
                          b'"paras":{"version": "v1.2",' \
                          b'"url": "https://10.1.1.1:8943/iodm/inner/v1.3.0/firmwarefiles/ca1d954771ae61e5098c7f83",' \
                          b'"file_size": 81362928,"access_token": "595124473f866b033dfa1f","expires": 86400,' \
                          b'"sign": "595124473f866b033dfa1f7e831c8c99a12f6143f392dfa996a819010842c99d"},' \
                          b'"service_id":"$ota","event_type":"firmware_upgrade",' \
                          b'"event_time":"20220411T124924Z"}],' \
                          b'"object_device_id":"' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'"}'
        _handle_on_message(message)

        # assert
        self.assertEqual(self.result_dict["url"],
                         "https://10.1.1.1:8943/iodm/inner/v1.3.0/firmwarefiles/ca1d954771ae61e5098c7f83")
        return
