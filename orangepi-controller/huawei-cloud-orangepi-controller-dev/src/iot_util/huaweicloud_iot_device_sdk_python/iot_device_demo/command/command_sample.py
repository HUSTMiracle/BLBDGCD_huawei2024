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
演示如何直接使用DeviceClient处理平台下发的命令
"""

import time
import logging

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.client.listener.command_listener \
    import CommandListener
from iot_device_sdk_python.iot_device import IotDevice
from iot_device_sdk_python.client.request.command_response import CommandRsp


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(threadName)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CommandSampleListener(CommandListener):
    """
    实现命令监听器的一个例子
    """
    def __init__(self, iot_device: IotDevice):
        """ 传入一个IotDevice实例 """
        self.device = iot_device

    def on_command(self, request_id: str, service_id: str, command_name: str, paras: dict):
        """
        命令处理方式
        :param request_id:      请求id
        :param service_id:      服务id
        :param command_name:    命令名
        :param paras:           命令参数
        """
        logger.info("on_command requestId: " + request_id)
        logger.info("begin to handle command")

        """ process code here """
        logger.info(str(paras))

        # 命令响应
        command_rsp = CommandRsp()
        command_rsp.result_code = CommandRsp.success_code()
        command_rsp.response_name = command_name
        command_rsp.paras = {"content": "Hello Huawei"}
        self.device.get_client().respond_command(request_id, command_rsp)

def run():
    # 替换为真正的接入地址、device id和密钥，参考readme_CN.md中“上传产品模型并注册设备”
    server_uri = "access address"
    port = 8883
    device_id = "your device id"
    secret = "your device secret"
    # iot平台的CA证书，用于服务端校验
    iot_ca_cert_path = "./resources/root.pem"

    connect_auth_info = ConnectAuthInfo()
    connect_auth_info.server_uri = server_uri
    connect_auth_info.port = port
    connect_auth_info.id = device_id
    connect_auth_info.secret = secret
    connect_auth_info.iot_cert_path = iot_ca_cert_path
    connect_auth_info.bs_mode = ConnectAuthInfo.BS_MODE_DIRECT_CONNECT

    client_conf = ClientConf(connect_auth_info)

    device = IotDevice(client_conf)

    # 设置监听器
    device.get_client().set_command_listener(CommandSampleListener(device))

    if device.connect() != 0:
        logger.error("init failed")
        return

    while True:
        time.sleep(5)

if __name__ == "__main__":
    run()
