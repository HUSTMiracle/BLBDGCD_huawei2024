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
演示如何直接使用DeviceClient进行消息透传
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(threadName)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def run():
    # 替换为真正的接入地址、device id和密钥，参考readme_CN.md中“上传产品模型并注册设备”
    server_uri = "access address"
    port = 8883
    device_id = "your device id"  # 填入从云平台获取的设备id
    secret = "your device secret"  # 填入从云平台获取的设备密钥
    iot_ca_cert_path = "iot cert path"
    # 自定义topic 以$oc/devices/{device_id}/user/开头，
    custom_topic = "$oc/devices/{device_id}/user/{topic}"
    # 自定义策略Topic 可以不以$oc开头， 例如：testdevicetopic
    custom_policy_topic = "{custom_policy_topic}"

    connect_auth_info = ConnectAuthInfo()
    connect_auth_info.server_uri = server_uri
    connect_auth_info.port = port
    connect_auth_info.id = device_id
    connect_auth_info.secret = secret
    connect_auth_info.iot_cert_path = iot_ca_cert_path
    connect_auth_info.bs_mode = ConnectAuthInfo.BS_MODE_DIRECT_CONNECT

    client_conf = ClientConf(connect_auth_info)

    device = IotDevice(client_conf)

    if device.connect() != 0:
        logger.error("init failed")
        return

    logger.info("begin report message")
    default_publish_listener = DefaultPublishActionListener()
    device_message = DeviceMessage()
    device_message.content = "Hello Huawei"

    device_message2 = DeviceMessage()
    device_message2.content = "Custom topic Message"

    device_message3 = DeviceMessage()
    device_message3.content = "Custom Policy topic Message"

    device_message4 = DeviceMessage()
    device_message4.content = "Custom Binary Message"
    # 定时上报消息
    while True:
        # 通过平台默认topic上报消息
        device.get_client().report_device_message(device_message,
                                                  default_publish_listener)
        time.sleep(1)
        payload = json.dumps(device_message2.to_dict())
        # 通过平台自定义topic上报消息
        device.get_client().publish_raw_message(RawMessage(custom_topic, payload, 1), default_publish_listener)
        time.sleep(1)
        payload = json.dumps(device_message3.to_dict())
        # 通过平台自定义策略中的topic上报消息
        device.get_client().publish_raw_message(RawMessage(custom_policy_topic, payload, 1), default_publish_listener)
        time.sleep(1)
        # 上报二进制码流
        payload = device_message4.content.encode()
        device.get_client().publish_raw_message(RawMessage(custom_policy_topic, payload, 1), default_publish_listener)
        time.sleep(5)


if __name__ == "__main__":
    run()
