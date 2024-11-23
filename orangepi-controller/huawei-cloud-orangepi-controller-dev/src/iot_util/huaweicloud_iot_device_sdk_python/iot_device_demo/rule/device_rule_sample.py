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


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(threadName)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CommandSampleListener(CommandListener):
    """
    实现命令监听器的一个例子
    """
    def __init__(self):
        super().__init__()

    def on_command(self, request_id: str, service_id: str, command_name: str, paras: dict):
        """
        命令处理方式
        :param request_id:      请求id
        :param service_id:      服务id
        :param command_name:    命令名
        :param paras:           命令参数
        """
        logger.info("on_command requestId: " + request_id)
        logger.info("on_command serviceId: " + service_id)
        logger.info("on_command commandName: " + command_name)
        logger.info("begin to handle command")

        """ process code here """
        logger.info(str(paras))

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
    connect_auth_info.enable_rule_manage = False

    client_conf = ClientConf(connect_auth_info)

    device = IotDevice(client_conf)
    device.get_client().set_command_listener(CommandSampleListener())

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
