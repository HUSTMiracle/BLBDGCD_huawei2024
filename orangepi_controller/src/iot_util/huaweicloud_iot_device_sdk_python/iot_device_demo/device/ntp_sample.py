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
演示时间同步功能
"""
import logging
import time

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.utils.iot_util import get_gmt_timestamp
from iot_device_sdk_python.timesync.time_sync_listener import TimeSyncListener
from iot_device_sdk_python.timesync.time_sync_service import TimeSyncService
from iot_device_sdk_python.iot_device import IotDevice

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(threadName)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TimeSyncSampleListener(TimeSyncListener):
    def on_time_sync_response(self, device_send_time: int, server_recv_time: int, server_send_time: int):
        device_recv_time = get_gmt_timestamp()
        now = (server_recv_time + server_send_time + device_recv_time - device_send_time) / 2
        print("now is " + str(now))


def run():
    # 替换为真正的接入地址、device id和密钥，参考readme_CN.md中“上传产品模型并注册设备”
    server_uri = "access address"
    port = 8883
    device_id = "your device id"
    secret = "your device secret"
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

    """ 设置时间同步服务 """
    time_sync_service: TimeSyncService = device.get_time_sync_service()
    time_sync_listener = TimeSyncSampleListener()
    time_sync_service.set_listener(time_sync_listener)

    if device.connect() != 0:
        logger.error("init failed")
        return

    # 请求时间同步
    time_sync_service.request_time_sync()

    while True:
        time.sleep(5)


if __name__ == "__main__":
    run()
