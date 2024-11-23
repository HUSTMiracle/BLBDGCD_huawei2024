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

from __future__ import absolute_import, annotations

import logging
from unittest import TestCase, mock
from unittest.mock import MagicMock, patch, PropertyMock

from paho.mqtt.client import MQTTMessage

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.client.listener.device_message_listener import DeviceMessageListener
from iot_device_sdk_python.client.listener.raw_device_message_listener import RawDeviceMessageListener
from iot_device_sdk_python.client.request.device_message import DeviceMessage
from iot_device_sdk_python.client.request.raw_device_message import RawDeviceMessage
from iot_device_sdk_python.iot_device import IotDevice

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s - %(threadName)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TestData:
    def __init__(self, payload: bytes | str, is_system_format: bool, is_sub_device_message: bool):
        if isinstance(payload, bytes):
            self.payload = payload
        else:
            self.payload = payload.encode('utf8')
        self.is_system_format = is_system_format
        self.is_sub_device_message = is_sub_device_message


def Any(cls):
    class ClassAny:
        def __eq__(self, other):
            return isinstance(other, cls)

    return ClassAny()


class TestMessageMethod(TestCase):
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
        iot_ca_cert_path = "./resources/SignRSAOVSSLCA2018.crt.pem"

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

    def _send_msg2device(self, payload: bytes):
        # test
        _get_connection = getattr(self.device.get_client(), "_get_connection")
        _connection = _get_connection()
        _get_paho_client = getattr(_connection, "_get_paho_client")
        _paho_client = _get_paho_client()
        _handle_on_message = getattr(_paho_client, "_handle_on_message")

        msg = MQTTMessage()
        msg.topic = b'$oc/devices/' \
                    + self.device_id.encode(encoding="utf-8") \
                    + b'/messages/down'
        msg.payload = payload

        _handle_on_message(msg)

    def test_receive_message(self):
        class TestMessageListener(DeviceMessageListener):
            def __init__(self, result: dict):
                self.receive_message_result = result

            def on_device_message(self, message: DeviceMessage):
                self.receive_message_result["device_id"] = message.device_id
                self.receive_message_result["content"] = message.content

        receive_message_result = dict()
        self.device.get_client().set_device_msg_listener(TestMessageListener(receive_message_result))

        payload = b'{"object_device_id": "' \
                  + self.device_id.encode(encoding="utf-8") \
                  + b'","name": "TestMessageName","id": "id","content": "Hello Huawei"}'
        self._send_msg2device(payload)

        # assert
        self.assertEqual(receive_message_result["device_id"], self.device_id)
        self.assertEqual(receive_message_result["content"], "Hello Huawei")
        return

    def message_handler_general_test(self, data: TestData):
        print(data.payload)
        device_message_listener_mock = MagicMock(spec=DeviceMessageListener)
        raw_device_message_listener_mock = MagicMock(spec=RawDeviceMessageListener)

        self.device.get_client().set_device_msg_listener(device_message_listener_mock)
        self.device.get_client().set_raw_device_msg_listener(raw_device_message_listener_mock)
        self.device.on_device_message = MagicMock()

        self._send_msg2device(data.payload)

        raw_device_message_listener_mock.on_raw_device_message.assert_called_once_with(Any(RawDeviceMessage))

        if data.is_system_format and not data.is_sub_device_message:
            device_message_listener_mock.on_device_message.assert_called_once_with(Any(DeviceMessage))
        else:
            device_message_listener_mock.on_device_message.assert_not_called()

        if data.is_system_format and data.is_sub_device_message:
            self.device.on_device_message.assert_called_with(Any(DeviceMessage))
        else:
            self.device.on_device_message.assert_not_called()

    def test_not_set_callback(self):
        """
        make sure it doesn't crash when both callback is null
        """
        self._send_msg2device("sdfdssdfdsffsd".encode("utf8"))

    def test_message_in_system_format(self):

        test_data = [
            TestData("""{"name":"1","id":"2",   "content":"3","object_device_id":null}""", True, False),
            TestData("""{"name": null,"id":"2", "content":"3","object_device_id":"1"}""", True, True),
            TestData("""{"name":"1","id":null,  "content":"3","object_device_id":"1"}""", True, True),
            TestData("""{"name":"1","id":"2",   "content":"3"}""", True, False),
            TestData("""{{"content":"3", "object_device_id": "{}"}}""".format(self.device_id), True, False)

        ]

        for d in test_data:
            self.message_handler_general_test(d)

    def test_message_in_non_system_format(self):
        test_data = [
            TestData("{\"name1\":\"1\",\"id\":\"2\",\"content\":\"3\",\"object_device_id\":null}", False, False),
            TestData("{\"content\":\"3\",\"object_device_id\":1,\"object_device_id22\":\"1\"}", False, True),
            TestData("ddf", False, True),
            TestData("{\"name\":[1],\"id\":null,\"content\":\"3\",\"object_device_id\":\"1\"}", False, False),
        ]
        for d in test_data:
            self.message_handler_general_test(d)

    def test_message_in_binary_format(self):
        test_data = [
            TestData(bytes([0x1, 0x2, 0x3, 0x4, 0x5]), False, False),
            TestData(bytes([107, 114, 91, 29]), False, False),
        ]

        for d in test_data:
            self.message_handler_general_test(d)

    def test_raw_message_payload_correctness(self):
        device_message_listener_mock = MagicMock(spec=DeviceMessageListener)
        raw_device_message_listener_mock = MagicMock(spec=RawDeviceMessageListener)

        self.device.get_client().set_device_msg_listener(device_message_listener_mock)
        self.device.get_client().set_raw_device_msg_listener(raw_device_message_listener_mock)
        self.device.on_device_message = MagicMock()

        test_data = [
            # pyload,         if on_device_message is called,  the expected DeviceMessage dict, None means no chekc
            ["abcdefghijklmn", False, None],
            ["{\"name\":\"1\",\"id\":\"2\",\"content\":\"3\",\"object_device_id\":null}", True, {
                "name": "1",
                "id": "2",
                "content": "3",
                "object_device_id": None
            }]
        ]
        on_device_msg_called_times = 0
        on_raw_device_msg_called_times = 0
        for test_data, is_system_format, exptecd_device_msg_dict in test_data:
            payload = test_data.encode('utf8')
            self._send_msg2device(payload)

            raw_device_message_listener_mock.on_raw_device_message.assert_called_with(Any(RawDeviceMessage))
            on_raw_device_msg_called_times += 1
            self.assertEqual(raw_device_message_listener_mock.on_raw_device_message.call_count,
                             on_raw_device_msg_called_times)

            args, kwargs = raw_device_message_listener_mock.on_raw_device_message.call_args
            raw_device_msg: RawDeviceMessage = args[0]
            self.assertEqual([raw_device_msg.payload, raw_device_msg.to_utf8_string()], [payload, test_data])

            if is_system_format:
                on_device_msg_called_times += 1
                self.assertEqual(device_message_listener_mock.on_device_message.call_count, on_device_msg_called_times)
                device_message_listener_mock.on_device_message.assert_called_with(Any(DeviceMessage))
                args, kwargs = device_message_listener_mock.on_device_message.call_args
                device_msg: DeviceMessage = args[0]
                if exptecd_device_msg_dict:
                    self.assertEqual(device_msg.to_dict(), exptecd_device_msg_dict)
            else:
                device_message_listener_mock.on_device_message.assert_not_called()
