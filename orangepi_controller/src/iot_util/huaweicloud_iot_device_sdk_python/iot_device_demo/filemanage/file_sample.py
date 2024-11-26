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
演示文件上传/下载功能
"""

import logging
import time
import os
from urllib.parse import urlparse

import requests

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.filemanager.file_manager_service import FileManagerService
from iot_device_sdk_python.filemanager.file_manager_listener import FileManagerListener
from iot_device_sdk_python.filemanager.url_info import UrlInfo
from iot_device_sdk_python.iot_device import IotDevice

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(threadName)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class FileManagerSampleListener(FileManagerListener):

    def __init__(self, file_manager: FileManagerService):
        self._file_manager = file_manager

    def on_upload_url(self, url_info: UrlInfo):
        """
        接收文件上传url
        :param url_info:   上传参数
        """
        if url_info.object_name not in self._file_manager.upload_file_dict.keys():
            raise RuntimeError("object_name: " + url_info.object_name + " has no related file_path")

        file_path = self._file_manager.upload_file_dict.get(url_info.object_name)
        if not os.path.isfile(file_path):
            raise RuntimeError("file_path: " + file_path + " is not file")

        data = open(file_path, 'rb').read()
        headers = {"Content-Type": "text/plain", "Host": urlparse(url_info.url).netloc}
        resp = requests.put(url=url_info.url, data=data, headers=headers)
        if resp.status_code == requests.codes.ok:
            # 成功时状态码为200
            logger.info("upload file success. url is: %s" % url_info.url)
            self._file_manager.report_upload_result(object_name=url_info.object_name,
                                                    result_code=0,
                                                    status_code=resp.status_code)
        else:
            logger.error("upload file fail, status code: %s" % str(resp.status_code))
            logger.error("response content: %s" % resp.text)
            self._file_manager.report_upload_result(object_name=url_info.object_name,
                                                    result_code=1,
                                                    status_code=resp.status_code)
        pass

    def on_download_url(self, url_info: UrlInfo):
        """
        接收文件下载url
        :param url_info:   下载参数
        """
        if url_info.object_name not in self._file_manager.download_file_dict.keys():
            raise RuntimeError("object_name: " + url_info.object_name + " has no related file_path")

        file_path = self._file_manager.download_file_dict.get(url_info.object_name)

        headers = {"Content-Type": "text/plain", "Host": urlparse(url_info.url).netloc}
        resp = requests.get(url=url_info.url, headers=headers)
        open(file_path, 'wb').write(resp.content)
        if resp.status_code == requests.codes.ok:
            logger.info("download file success.")
            self._file_manager.report_download_result(object_name=url_info.object_name,
                                                      result_code=0,
                                                      status_code=resp.status_code)
        else:
            logger.error("download file fail, status code: %s" % str(resp.status_code))
            logger.error("response content: %s" % resp.text)
            self._file_manager.report_download_result(object_name=url_info.object_name,
                                                      result_code=1,
                                                      status_code=resp.status_code)
        pass


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

    client_conf = ClientConf(connect_auth_info)

    device = IotDevice(client_conf)

    """ 设置文件管理监听器 """
    file_manager: FileManagerService = device.get_file_manager_service()
    file_manager_listener = FileManagerSampleListener(file_manager)
    file_manager.set_listener(file_manager_listener)

    if device.connect() != 0:
        logger.error("init failed")
        return

    """ 文件上传 """
    upload_file_path = os.path.dirname(__file__) + r'/download/upload_test.txt'
    file_name = "upload_test.txt"
    file_manager.upload_file(file_name=file_name, file_path=upload_file_path)

    # 10s后将刚刚上传的upload_test.txt下载下来，保存到download.txt
    time.sleep(10)

    # """ 文件下载 """
    download_file_path = os.path.dirname(__file__) + r'/download/download.txt'
    file_manager.download_file(file_name=file_name, file_path=download_file_path)

    while True:
        time.sleep(5)


if __name__ == "__main__":
    run()
