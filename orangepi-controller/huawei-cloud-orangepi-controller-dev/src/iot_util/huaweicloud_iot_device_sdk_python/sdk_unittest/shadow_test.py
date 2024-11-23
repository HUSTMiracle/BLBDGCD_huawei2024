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
from typing import List

from paho.mqtt.client import MQTTMessage

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.client.request.shadow_data import ShadowData
from iot_device_sdk_python.iot_device import IotDevice
from iot_device_sdk_python.client.listener.device_shadow_listener import DeviceShadowListener


class TestShadowMethod(TestCase):

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

        self.device = IotDevice(client_conf)

        self.device.connect()
        return

    def test_receive_shadow(self):

        class TestShadowListener(DeviceShadowListener):

            def __init__(self, result: dict):
                self.receive_command_result = result

            def on_shadow_get(self, request_id: str, object_device_id: str, shadow: List[ShadowData]):
                _reported = dict()
                _reported["properties"] = shadow[0].reported.properties
                _reported["event_time"] = shadow[0].reported.event_time
                self.receive_command_result["reported"] = _reported
                _desired = dict()
                _desired["properties"] = shadow[0].desired.properties
                _desired["event_time"] = shadow[0].desired.event_time
                self.receive_command_result["desired"] = _desired

        receive_shadow_result = dict()
        self.device.get_client().set_device_shadow_listener(TestShadowListener(receive_shadow_result))

        # test
        _get_connection = getattr(self.device.get_client(), "_get_connection")
        _connection = _get_connection()
        _get_paho_client = getattr(_connection, "_get_paho_client")
        _paho_client = _get_paho_client()
        _handle_on_message = getattr(_paho_client, "_handle_on_message")
        message = MQTTMessage()
        req_id = "123456"

        message.topic = b'$oc/devices/' \
                        + self.device_id.encode(encoding="utf-8") \
                        + b'/sys/shadow/get/response/request_id=' \
                        + req_id.encode(encoding="utf-8")
        message.payload = b'{"object_device_id": "' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'","shadow": [{"service_id": "WaterMeter",' \
                            b'"desired": {"properties": {"temperature": "60"},"event_time": "20151212T121212Z"},' \
                            b'"reported": {"properties": {"temperature": "60"},"event_time": "20151212T121212Z"},' \
                            b'"version": 1}]}'
        _handle_on_message(message)

        reported = {
            "properties": {
                "temperature": "60"
            },
            "event_time": "20151212T121212Z"
        }
        desired = {
            "properties": {
                "temperature": "60"
            },
            "event_time": "20151212T121212Z"
        }
        # assert
        self.assertEqual(receive_shadow_result["reported"], reported)
        self.assertEqual(receive_shadow_result["desired"], desired)
        return

