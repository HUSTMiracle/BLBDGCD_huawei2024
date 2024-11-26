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

"""
演示如何直接使用DeviceClient进行消息下发
"""
import json
import logging
import time

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.client.listener.default_publish_action_listener import DefaultPublishActionListener
from iot_device_sdk_python.client.listener.raw_device_message_listener import RawDeviceMessageListener
from iot_device_sdk_python.client.request.device_message import DeviceMessage
from iot_device_sdk_python.client.request.raw_device_message import RawDeviceMessage
from iot_device_sdk_python.iot_device import IotDevice
from iot_device_sdk_python.transport.raw_message import RawMessage
from iot_device_sdk_python.transport.connect_listener import ConnectListener

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

    # 自定义topic订阅需放在链接建立成功后，防止断线重连后没有订阅topic
    def connect_complete(self, reconnect: bool, server_uri: str):
        """
        连接成功通知，如果是断链重连的情景，重连成功会重新订阅topic

        Args:
            reconnect:   是否为重连（当前此参数没有作用）
            server_uri:  服务端地址
        """
        # 自定义topic 以$oc/devices/{device_id}/user/开头，
        custom_topic = "$oc/devices/{device_id}/user/{topic}"
        # 自定义策略Topic 可以不以$oc开头， 例如：testdevicetopic
        custom_policy_topic = "{custom_topic}"
        # 使用自定义topic设置监听器接收平台下行消息
        self.device.get_client().subscribe_topic(custom_topic, 1, RawDeviceMsgListener())
        # 使用自定义策略topic设置监听器接收平台下行消息
        self.device.get_client().subscribe_topic(custom_policy_topic, 1, RawDeviceMsgListener())
        logger.info("connect success. server uri is " + server_uri)


class RawDeviceMsgListener(RawDeviceMessageListener):
    def on_raw_device_message(self, message: RawDeviceMessage):
        """
        处理平台下发的设备消息
        :param message:     设备消息内容
        """
        device_msg = message.to_device_message()
        if device_msg:
            print("on_device_message got system format:", message.payload)
        else:
            print("on_device_message:", message.payload)

        """ code here """
        pass

    def on_message_received(self, message: RawMessage):
        """
        处理平台自定义topic以及自定义策略topic下发的设备消息
        param message:  设备消息内容
        """
        topic = message.topic
        print("get message: " + str(message.payload) + " from topic: " + topic)


def run():
    # 替换为真正的接入地址、device id和密钥，参考readme_CN.md中“上传产品模型并注册设备”
    server_uri = "access address"
    port = 8883
    device_id = "your device id"
    secret = "your secret"
    iot_ca_cert_path = "iot ca path"


    connect_auth_info = ConnectAuthInfo()
    connect_auth_info.server_uri = server_uri
    connect_auth_info.port = port
    connect_auth_info.id = device_id
    connect_auth_info.secret = secret
    connect_auth_info.iot_cert_path = iot_ca_cert_path
    connect_auth_info.bs_mode = ConnectAuthInfo.BS_MODE_DIRECT_CONNECT
    connect_auth_info.reconnect_on_failure = True

    client_conf = ClientConf(connect_auth_info)

    device = IotDevice(client_conf)

    # 使用默认topic设置监听器接收平台下行消息
    device.get_client().set_raw_device_msg_listener(RawDeviceMsgListener())
    device.get_client().add_connect_listener(CustomConnectListener(device))
    if device.connect() != 0:
        logger.error("init failed")
        return



if __name__ == "__main__":
    run()
