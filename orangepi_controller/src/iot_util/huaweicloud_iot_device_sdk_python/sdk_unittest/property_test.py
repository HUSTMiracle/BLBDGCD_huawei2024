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
from typing import List
from unittest import TestCase, mock

from paho.mqtt.client import MQTTMessage

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.client.request.service_property import ServiceProperty
from iot_device_sdk_python.iot_device import IotDevice
from iot_device_sdk_python.client.listener.property_listener import PropertyListener


class TestPropertyMethod(TestCase):

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

    def test_receive_property_set(self):

        class TestPropertyListener(PropertyListener):
            def __init__(self, result: dict):
                self.receive_command_result = result

            def on_property_set(self, request_id: str, services: List[ServiceProperty]):
                self.receive_command_result["services"] = services

            def on_property_get(self, request_id: str, service_id: str):
                pass

        receive_property_result = dict()
        self.device.get_client().set_properties_listener(TestPropertyListener(receive_property_result))

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
                        + b'/sys/properties/set/request_id=' \
                        + req_id.encode(encoding="utf-8")
        message.payload = b'{"object_device_id": "' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'","services": [{"service_id": "Battery", "properties": {"level": 80}}]}'
        _handle_on_message(message)

        # assert
        services_result: List[ServiceProperty] = receive_property_result.get("services")
        self.assertEqual(services_result[0].service_id, "Battery")
        self.assertEqual(services_result[0].properties.get("level"), 80)
        return

    def test_receive_property_get(self):

        class TestPropertyListener(PropertyListener):
            def __init__(self, result: dict):
                self.receive_command_result = result

            def on_property_set(self, request_id: str, services: List[ServiceProperty]):
                pass

            def on_property_get(self, request_id: str, service_id: str):
                self.receive_command_result["service_id"] = service_id

        receive_property_result = dict()
        self.device.get_client().set_properties_listener(TestPropertyListener(receive_property_result))

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
                        + b'/sys/properties/get/request_id=' \
                        + req_id.encode(encoding="utf-8")
        message.payload = b'{"object_device_id": "' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'","service_id": "TestPropertyGet"}'
        _handle_on_message(message)

        # assert
        services_result = receive_property_result.get("service_id")
        self.assertEqual(services_result, "TestPropertyGet")
        return

