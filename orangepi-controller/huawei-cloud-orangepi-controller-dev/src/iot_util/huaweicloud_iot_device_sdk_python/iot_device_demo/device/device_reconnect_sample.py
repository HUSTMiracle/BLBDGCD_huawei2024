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
import logging
import time
from collections import deque
from typing import Optional

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.client.request.device_message import DeviceMessage
from iot_device_sdk_python.iot_device import IotDevice
from iot_device_sdk_python.transport.connect_listener import ConnectListener
from iot_device_sdk_python.client.listener.default_publish_action_listener import DefaultPublishActionListener

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(threadName)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CustomConnectListener(ConnectListener):

    def __init__(self, iot_device: IotDevice):
        """ 传入一个IotDevice实例 """
        self.device = iot_device

    def connection_lost(self, cause: str):
        """
        连接丢失通知

        Args:
            cause:   连接丢失原因
        """
        logger.warning("connection lost. cause: " + cause)
        logger.warning("you can define reconnect in this method.")

    def connect_complete(self, reconnect: bool, server_uri: str):
        """
        连接成功通知，如果是断链重连的情景，重连成功会上报断链的时间戳

        Args:
            reconnect:   是否为重连（当前此参数没有作用）
            server_uri:  服务端地址
        """
        logger.info("connect success. server uri is " + server_uri)


def run():
    # 替换为真正的接入地址、device id和密钥，参考readme_CN.md中“上传产品模型并注册设备”
    server_uri = "access address"
    port = 1883
    device_id = "your device id"
    secret = "your secret"

    connect_auth_info = ConnectAuthInfo()
    connect_auth_info.server_uri = server_uri
    connect_auth_info.port = port
    connect_auth_info.id = device_id
    connect_auth_info.secret = secret
    connect_auth_info.check_timestamp = "0"
    connect_auth_info.bs_mode = ConnectAuthInfo.BS_MODE_DIRECT_CONNECT
    connect_auth_info.reconnect_on_failure = True
    connect_auth_info.min_backoff = 1 * 1000
    connect_auth_info.max_backoff = 30 * 1000
    connect_auth_info.max_buffer_message = 100

    client_conf = ClientConf(connect_auth_info)

    device = IotDevice(client_conf)
    if device.connect() != 0:
        logger.error("init failed")
        return

    device.get_client().add_connect_listener(CustomConnectListener(device))
    logger.info("begin report message")
    device_message = DeviceMessage()
    device_message.content = "Hello Huawei"
    # 定时上报消息
    while True:
        device.get_client().report_device_message(device_message, DefaultPublishActionListener())
        time.sleep(10)


if __name__ == "__main__":
    run()