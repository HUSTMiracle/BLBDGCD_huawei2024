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
from iot_device_sdk_python.timesync.time_sync_listener import TimeSyncListener
from iot_device_sdk_python.timesync.time_sync_service import TimeSyncService


class TestTimeSyncMethod(TestCase):

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

    def test_receive_time_sync_response(self):

        class TimeSyncSampleListener(TimeSyncListener):
            def __init__(self, result_dict: dict):
                self.result_dict = result_dict

            def on_time_sync_response(self, device_send_time: int, server_recv_time: int, server_send_time: int):
                self.result_dict["device_send_time"] = device_send_time
                self.result_dict["server_recv_time"] = server_recv_time
                self.result_dict["server_send_time"] = server_send_time

        response_dict = dict()
        # 设置时间同步服务
        time_sync_service: TimeSyncService = self.device.get_time_sync_service()
        time_sync_listener = TimeSyncSampleListener(response_dict)
        time_sync_service.set_listener(time_sync_listener)

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
                          b'"paras":{"device_send_time":1649681364061,"server_recv_time":1649681364120,' \
                          b'"server_send_time":1649681364120},' \
                          b'"service_id":"$time_sync","event_type":"time_sync_response",' \
                          b'"event_time":"20220411T124924Z"}],' \
                          b'"object_device_id":"' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'"}'
        _handle_on_message(message)

        # assert
        self.assertEqual(response_dict["device_send_time"], 1649681364061)
        self.assertEqual(response_dict["server_recv_time"], 1649681364120)
        self.assertEqual(response_dict["server_send_time"], 1649681364120)
        return

