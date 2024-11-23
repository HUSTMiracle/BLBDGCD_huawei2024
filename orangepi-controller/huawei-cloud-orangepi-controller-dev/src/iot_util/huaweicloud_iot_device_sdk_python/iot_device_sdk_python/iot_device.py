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
from typing import List
import logging

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.client.device_client import DeviceClient
from iot_device_sdk_python.filemanager.file_manager_service import FileManagerService
from iot_device_sdk_python.ota.ota_service import OTAService
from iot_device_sdk_python.rule.model.action_handler import ActionHandler
from iot_device_sdk_python.rule.rule_manage_service import RuleManageService
from iot_device_sdk_python.service.abstract_device import AbstractDevice
from iot_device_sdk_python.service.abstract_service import AbstractService
from iot_device_sdk_python.timesync.time_sync_service import TimeSyncService


class IotDevice(AbstractDevice):
    """
    IOT设备类,SDK的入口类,提供两种使用方式:

    1、面向物模型编程:根据物模型实现设备服务,SDK自动完成设备和平台之间的通讯。这种方式简单易用,适合大多数场景。
    例子：请参考/iot_device_demo/smoke_detector.py

    2、面向通讯接口编程:获取设备的客户端，直接和平台进行通讯。这种方式更复制也更灵活。
    例子:device = IotDevice(...)

         device.get_client().set_command_listener(...)

         device.get_client().report_device_message(...)
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, client_conf: ClientConf):
        self._logger.debug("begin init IotDevice")
        super().__init__(client_conf)

    def connect(self):
        """
        初始化，创建到平台的连接

        Returns:
            int: 如果连接成功,返回0;否则返回-1
        """
        self._logger.debug("begin connect")
        return super().connect()

    def get_client(self) -> DeviceClient:
        """
        获取设备客户端。获取到设备客户端后，可以直接调用客户端提供的消息、属性、命令等接口。

        Returns:
            DeviceClient: 设备客户端实例
        """
        self._logger.debug("get DeviceClient")
        return super().get_client()

    def get_ota_service(self) -> OTAService:
        """
        获取OTA服务

        Returns:
            OTAService: OTA服务
        """
        self._logger.debug("get OTAService")
        return super().get_ota_service()

    def get_time_sync_service(self) -> TimeSyncService:
        """
        获取时间同步服务

        Returns:
            TimeSyncService: 时间同步服务
        """
        self._logger.debug("get TimeSyncService")
        return super().get_time_sync_service()

    def get_file_manager_service(self) -> FileManagerService:
        """
        获取文件管理服务

        Returns:
            FileManagerService: 文件管理服务
        """
        self._logger.debug("get FileManagerService")
        return super().get_file_manager_service()

    def get_rule_manage_service(self) -> RuleManageService:
        self._logger.debug("get RuleManageService")
        return super().get_rule_manage_service()

    def destroy(self):
        """
        释放连接
        """
        self._logger.debug("destroy connection")
        super().destroy()

    @staticmethod
    def create_by_secret(server_uri: str, port: int, device_id: str, secret: str, iot_cert_file: str = ""):
        """
        使用密钥创建设备

        Args:
            server_uri:      平台访问地址,比如ssl://iot-acc.cn-north-4.myhuaweicloud.com
            port:            端口，比如:8883
            device_id:       设备id
            secret:          设备密码
            iot_cert_file:   iot平台的ca证书,用于双向校验时设备侧校验平台。端口为8883时,此项必填;端口为1883时,此项可不填。
        """
        # 组装ConnectAuthInfo
        connect_auth_info = ConnectAuthInfo()
        connect_auth_info.server_uri = server_uri
        connect_auth_info.port = port
        connect_auth_info.id = device_id
        connect_auth_info.secret = secret
        connect_auth_info.iot_cert_path = iot_cert_file

        client_conf = ClientConf(connect_auth_info)
        return IotDevice(client_conf)

    @staticmethod
    def create_by_certificate(server_uri: str, port: int, device_id: str, cert_path: str, key_path: str,
                              iot_cert_file: str):
        """
        使用证书创建设备

        Args:
            server_uri:      平台访问地址，比如ssl://iot-acc.cn-north-4.myhuaweicloud.com
            port:            端口，比如:8883
            device_id:       设备id
            cert_path:       x509证书的pem
            key_path:        x509证书的key
            iot_cert_file:   iot平台的ca证书，用于双向校验时设备侧校验平台
        """
        # 组装ConnectAuthInfo
        connect_auth_info = ConnectAuthInfo()
        connect_auth_info.server_uri = server_uri
        connect_auth_info.port = port
        connect_auth_info.id = device_id
        connect_auth_info.cert_path = cert_path
        connect_auth_info.key_path = key_path
        connect_auth_info.iot_cert_path = iot_cert_file

        client_conf = ClientConf(connect_auth_info)
        return IotDevice(client_conf)

    def add_service(self, service_id: str, device_service: AbstractService):
        """
        添加服务。用户基于AbstractService定义自己的设备服务，并添加到设备。

        Args:
            service_id:      服务id，要和设备模型定义一致
            device_service:  服务实例
        """
        self._logger.debug("add Service")
        super().add_service(service_id, device_service)

    def get_service(self, service_id: str):
        """
        通过服务id获取服务实例

        Args:
            service_id:  服务id
        Returns:
            AbstractService: 服务实例。若服务不存在，则返回None
        """
        self._logger.debug("get Service")
        return super().get_service(service_id)

    def fire_properties_changed(self, service_id: str, properties: list):
        """
        触发属性变化，SDK会上报变化的属性

        Args:
            service_id: 服务id
            properties: 属性列表
        """
        self._logger.debug("fire properties changed")
        super().fire_properties_changed(service_id, properties)

    def fire_services_changed(self, service_ids: List[str]):
        """
        触发多个服务的属性变化，SDK自动上报变化的属性到平台

        Args:
            service_ids: 发生变化的服务id列表
        """
        self._logger.debug("fire services changed")
        super().fire_services_changed(service_ids)

