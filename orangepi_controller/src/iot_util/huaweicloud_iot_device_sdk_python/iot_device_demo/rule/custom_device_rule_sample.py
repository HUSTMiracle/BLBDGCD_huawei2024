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
演示如何直接使用DeviceClient获取端侧规则并执行
通过命令回调显示规则执行结果
"""

from typing import List
import time
import uuid
import logging

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.client.request.shadow_data import ShadowData
from iot_device_sdk_python.iot_device import IotDevice
from iot_device_sdk_python.client.request.service_property import ServiceProperty
from iot_device_sdk_python.client import iot_result
from iot_device_sdk_python.client.listener.command_listener import CommandListener
from iot_device_sdk_python.rule.model.action_handler import ActionHandler
from iot_device_sdk_python.rule.model.actions import Action
from iot_device_sdk_python.client.request.command import Command


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(threadName)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ActionHandlerListener(ActionHandler):
    """
    实现自定义端侧规则处理的一个例子
    """
    def __init__(self):
        super().__init__()

    def handle_rule_action(self, action_list: List[Action]):
        """
        命令处理方式
        :param action_list:      端侧规则动作信息
        """
        for action in action_list:
            device_id = action.device_id
            command: Command = action.command
            logger.info("on_action deviceId: " + device_id)
            logger.info("on_action serviceId: " + command.service_id)
            logger.info("on_action param: " + str(command.paras))

def run():
    # 替换为真正的接入地址、device id和密钥，参考readme_CN.md中“上传产品模型并注册设备”
    server_uri = "access address"
    port = 8883
    device_id = "your device id"
    secret = "your device secret"
    iot_ca_cert_path = "../resources/root.pem"

    connect_auth_info = ConnectAuthInfo()
    connect_auth_info.server_uri = server_uri
    connect_auth_info.port = port
    connect_auth_info.id = device_id
    connect_auth_info.secret = secret
    connect_auth_info.iot_cert_path = iot_ca_cert_path
    connect_auth_info.bs_mode = ConnectAuthInfo.BS_MODE_DIRECT_CONNECT
    connect_auth_info.enable_rule_manage = True

    client_conf = ClientConf(connect_auth_info)

    device = IotDevice(client_conf)
    device.get_client().set_rule_action_handler(ActionHandlerListener())

    if device.connect() != 0:
        logger.error("init failed")
        return

    time.sleep(5)
    logger.info("begin to report properties")
    # 按照产品模型设置属性
    service_property = ServiceProperty()
    service_property.service_id = "smokeDetector"
    service_property.properties = {"temperature": 10}
    # 组装成列表的形式
    services = [service_property]

    # 上报设备属性
    device.get_client().report_properties(services)
    while True:
        time.sleep(5)


if __name__ == "__main__":
    run()
