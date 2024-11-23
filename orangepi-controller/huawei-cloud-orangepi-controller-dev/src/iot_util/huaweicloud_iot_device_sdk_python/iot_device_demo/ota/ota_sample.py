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
OTA sample,用来演示如何实现设备升级（此例子不再使用）
使用方法:用户在平台上创建升级任务后,修改main函数里设备参数后启动本例,即可看到设备收到升级通知,并下载升级包进行升级,
并上报升级结果。在平台上可以看到升级结果
"""

import time
import os
import logging

import requests

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.ota.ota_listener import OTAListener
from iot_device_sdk_python.iot_device import IotDevice
from iot_device_sdk_python.ota.ota_query_version import OTAQueryVersion
from iot_device_sdk_python.ota.ota_service import OTAService
from iot_device_sdk_python.ota.ota_package_info import OTAPackageInfo
from iot_device_sdk_python.ota.ota_package_info_v2 import OTAPackageInfoV2
from iot_device_sdk_python.utils.iot_util import sha256_hash_from_file
from typing import Union

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(threadName)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")


class OTASampleListener(OTAListener):
    """
    一个实现OTA监听器的例子
    """

    def __init__(self, ota_service: OTAService):
        self._logger = logging.getLogger(__name__)
        self.fw_version = "v1.0.0"  # 自行修改为实际值
        self.sw_version = "v1.0.0"  # 自行修改为实际值
        self.ota_service = ota_service

    def on_query_version(self, queryInfo: OTAQueryVersion):
        """
        接收查询版本通知
        """
        self._logger.info("OTASampleListener on_query_version, queryInfo=%s", str(queryInfo.to_dict()))
        self.ota_service.report_version(self.fw_version, self.sw_version)

    def on_receive_package_info(self, pkg: Union[OTAPackageInfo, OTAPackageInfoV2]):
        """
        接收新版本通知
        :param pkg:     新版本包信息
        """
        self._logger.info("OTASampleListener on_receive_package_info, pkg=%s", str(pkg.to_dict()))
        if self.pre_check(pkg) != 0:
            self._logger.error("pre_check failed")
            return
        # 下载包并升级
        self.download_package(pkg)

    def pre_check(self, pkg: Union[OTAPackageInfo, OTAPackageInfoV2]):
        """
        对新版本包信息进行检查
        :param pkg:     新版本包信息
        :return:    如果允许升级，返回0；返回非0表示不允许升级
        """
        # TODO 对版本号、剩余空间、剩余电量、信号质量等进行检查、如果不允许升级，上报OTAService中定义的错误码或者自定义错误码，返回-1
        return 0

    def on_upgrade_success(self, pkg: Union[OTAPackageInfo, OTAPackageInfoV2], sign: str):
        self._logger.info("download package success")
        # 校验下载的升级包
        if self.check_package(pkg, sign) != 0:
            return

        # 安装升级包
        if self.install_package() != 0:
            return

        # 上报升级成功，注意版本号要携带更新后的版本号，否则平台会认为升级失败
        self.ota_service.report_ota_status(self.ota_service.OTA_CODE_SUCCESS, pkg.version, 100,
                                           "upgrade success")
        self._logger.info("ota upgrade ok")

    def on_upgrade_failure(self):
        self._logger.error("download package failed")

    def check_package(self, pkg: Union[OTAPackageInfo, OTAPackageInfoV2], sign: str):
        """
        校验升级包
        :param pkg:     新版本包信息
        :param sign:    str
        :return:    0表示校验成功；非0表示校验失败
        """
        if isinstance(pkg, OTAPackageInfo) and sign != pkg.sign:
            self._logger.error(
                "check package fail: current file sha256 %s , target file sha256 %s" % (sign, pkg.sign))
            return -1
        # TODO 增加其他校验
        return 0

    def install_package(self):
        """
        安装升级包，需要用户实现
        :return:    0表示安装成功；非0表示安装失败
        """
        # TODO 安装升级包，用户实现
        return 0

    def download_package(self, pkg: Union[OTAPackageInfo, OTAPackageInfoV2]):
        """
        下载包，这里的例子是下载一个txt文件
        # TODO 下载升级包
        """
        path = os.path.dirname(os.path.realpath("__file__")) + r'/download'
        if not os.path.exists(path):  # 看是否有该文件夹，没有则创建文件夹
            os.makedirs(path)

        headers = {}
        if isinstance(pkg, OTAPackageInfo):
            headers = {"Authorization": "Bearer " + pkg.access_token}

        response = requests.get(pkg.url, headers=headers, verify=False, stream=True)

        chunk_size = 1024  # 每次下载的数据大小
        try:
            if response.status_code == 200:
                # 下载文件总大小
                content_size = int(response.headers["content-length"])

                filepath = path + "/test.txt"
                size = 0
                with open(filepath, 'wb') as file:  # 显示进度条
                    for data in response.iter_content(chunk_size=chunk_size):
                        file.write(data)
                        size += len(data)
                if size == content_size:
                    file_sha256_hash: str = sha256_hash_from_file(filepath)
                    self.on_upgrade_success(pkg, file_sha256_hash)
        except Exception as e:
            self.on_upgrade_failure()
            raise e


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

    """ OTA监听器设置 """
    ota_service: OTAService = device.get_ota_service()
    ota_service_listener = OTASampleListener(ota_service)
    ota_service.set_ota_listener(ota_service_listener)

    if device.connect() != 0:
        return

    while True:
        time.sleep(5)


if __name__ == "__main__":
    run()
