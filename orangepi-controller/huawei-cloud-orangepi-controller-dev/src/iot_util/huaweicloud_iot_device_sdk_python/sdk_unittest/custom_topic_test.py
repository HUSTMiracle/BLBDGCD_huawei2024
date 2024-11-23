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
from iot_device_sdk_python.transport.raw_message_listener import RawMessageListener
from iot_device_sdk_python.transport.raw_message import RawMessage


class TestCustomTopicMethod(TestCase):

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

    def test_receive_custom_topic_message(self):

        class MyMessageListener(RawMessageListener):
            """
            自定义topic的例子
            """
            def __init__(self, result_dict: dict):
                self.result_dict = result_dict

            def on_message_received(self, message: RawMessage):
                self.result_dict["payload"] = str(message.payload)

        payload_dict = dict()
        my_topic = "$oc/devices/6109fd1da42d680286bb1ff3_123456/user/topictest"
        self.device.get_client().subscribe_topic(my_topic, 1, MyMessageListener(payload_dict))

        # test
        _get_connection = getattr(self.device.get_client(), "_get_connection")
        _connection = _get_connection()
        _get_paho_client = getattr(_connection, "_get_paho_client")
        _paho_client = _get_paho_client()
        _handle_on_message = getattr(_paho_client, "_handle_on_message")
        message = MQTTMessage()
        message.topic = my_topic.encode(encoding="utf-8")
        message.payload = b'{"object_device_id": "' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'","paras": {"value": "123"}}'
        _handle_on_message(message)

        # assert
        self.assertEqual(payload_dict["payload"], str(message.payload))
        return

