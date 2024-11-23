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
from iot_device_sdk_python.filemanager.file_manager_listener import FileManagerListener
from iot_device_sdk_python.filemanager.file_manager_service import FileManagerService
from iot_device_sdk_python.filemanager.url_info import UrlInfo
from iot_device_sdk_python.iot_device import IotDevice


class FileManagerTestListener(FileManagerListener):

    def __init__(self, result_dict: dict):
        self.result_dict = result_dict

    def on_upload_url(self, url_info: UrlInfo):
        """
        接收文件上传url
        :param url_info:   上传参数
        """
        self.result_dict["upload_object_name"] = url_info.object_name

    def on_download_url(self, url_info: UrlInfo):
        """
        接收文件下载url
        :param url_info:   下载参数
        """
        self.result_dict["download_object_name"] = url_info.object_name


class TestFileManagerMethod(TestCase):

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
        # 设置文件管理监听器
        file_manager: FileManagerService = self.device.get_file_manager_service()
        file_manager_listener = FileManagerTestListener(self.result_dict)
        file_manager.set_listener(file_manager_listener)

        self.device.connect()
        return

    def test_receive_upload_url(self):
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
                          b'"paras":{"url":"https://bucket.obs.cn-north-4.com/device_file/",' \
                          b'"bucket_name":"bucket",' \
                          b'"object_name": "c6b39067b0325db34663d3ef421a42f6_12345678_a.jpg","expire":3600,' \
                          b'"file_attributes": {"hash_code": "58059181f378062f9b446e884362a526","size": 1024}},' \
                          b'"service_id":"$file_manager","event_type":"get_upload_url_response",' \
                          b'"event_time":"20220411T124924Z"}],' \
                          b'"object_device_id":"' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'"}'
        _handle_on_message(message)

        # assert
        self.assertEqual(self.result_dict["upload_object_name"], "c6b39067b0325db34663d3ef421a42f6_12345678_a.jpg")
        return

    def test_receive_download_url(self):
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
                          b'"paras":{"url":"https://bucket.obs.cn-north-4.com/device_file",' \
                          b'"bucket_name":"bucket",' \
                          b'"object_name": "c6b39067b0325db34663d3ef421a42f6_12345678_a.jpg","expire":3600,' \
                          b'"file_attributes": {"hash_code": "58059181f378062f9b446e884362a526","size": 1024}},' \
                          b'"service_id":"$file_manager","event_type":"get_download_url_response",' \
                          b'"event_time":"20220411T124924Z"}],' \
                          b'"object_device_id":"' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'"}'
        _handle_on_message(message)

        # assert
        self.assertEqual(self.result_dict["download_object_name"], "c6b39067b0325db34663d3ef421a42f6_12345678_a.jpg")
        return

