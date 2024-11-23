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
from typing import List, Optional
import logging

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.device_client import DeviceClient
from iot_device_sdk_python.client.mqtt_connect_conf import MqttConnectConf
from iot_device_sdk_python.devicelog.listener.default_conn_action_log_listener import DefaultConnActionLogListener
from iot_device_sdk_python.ota.ota_service import OTAService
from iot_device_sdk_python.service.abstract_service import AbstractService
from iot_device_sdk_python.client.request.command import Command
from iot_device_sdk_python.client.request.command_response import CommandRsp
from iot_device_sdk_python.client.request.props_set import PropSet
from iot_device_sdk_python.client.request.props_get import PropsGet
from iot_device_sdk_python.client.request.service_property import ServiceProperty
from iot_device_sdk_python.client.iot_result import IotResult
from iot_device_sdk_python.service.i_service import IService
from iot_device_sdk_python.client.iot_result import SUCCESS
from iot_device_sdk_python.client.request.device_events import DeviceEvents
from iot_device_sdk_python.client.request.device_event import DeviceEvent
from iot_device_sdk_python.client.request.device_message import DeviceMessage
from iot_device_sdk_python.timesync.time_sync_service import TimeSyncService
from iot_device_sdk_python.filemanager.file_manager_service import FileManagerService
from iot_device_sdk_python.devicelog.device_log_service import DeviceLogService
from iot_device_sdk_python.devicelog.listener.default_conn_log_listener import DefaultConnLogListener
from iot_device_sdk_python.utils.iot_util import get_event_time
from iot_device_sdk_python.rule.rule_manage_service import RuleManageService
from iot_device_sdk_python.rule.model.action_handler import ActionHandler
from iot_device_sdk_python.rule.model.actions import Action


class AbstractDevice:
    """
    抽象设备类
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, client_conf: ClientConf):
        connect_auth_info = client_conf.connect_auth_info
        mqtt_connect_conf = client_conf.mqtt_connect_conf

        if mqtt_connect_conf is None:
            mqtt_connect_conf = MqttConnectConf()
        self._client: DeviceClient = DeviceClient(connect_auth_info, mqtt_connect_conf, self)

        self._device_id = connect_auth_info.id

        self._services = dict()
        self._ota_service: Optional[OTAService] = None
        self._file_manager: Optional[FileManagerService] = None
        self._time_sync_service: Optional[TimeSyncService] = None
        self._device_log_service: Optional[DeviceLogService] = None
        self._rule_manage_service: Optional[RuleManageService] = None

        self._init_sys_services()

    def _init_sys_services(self):
        """
        初始化系统默认service，系统service以$作为开头。
        当前系统默认的服务有：OTA，时间同步服务，文件管理服务（文件上传/下载），设备日志服务
        """
        self._ota_service = OTAService()
        self.add_service(service_id="$ota", device_service=self._ota_service)
        self._time_sync_service = TimeSyncService()
        self.add_service(service_id="$time_sync", device_service=self._time_sync_service)
        self._file_manager = FileManagerService()
        self.add_service(service_id="$file_manager", device_service=self._file_manager)
        self._device_log_service = DeviceLogService()
        self.add_service(service_id="$log", device_service=self._device_log_service)
        self._rule_manage_service = RuleManageService(self._client)
        self.add_service(service_id="$device_rule", device_service=self._rule_manage_service)

    def connect(self):
        """
        初始化，创建到平台的连接

        Returns:
            int: 如果连接成功，返回0；否则返回-1
        """
        # 如果日志上报开关是关闭状态或者已过日志收集结束时间，则取消上报设备连接状态相关日志
        # TODO 需要优化
        if self._device_log_service and self._device_log_service.can_report_log():
            default_conn_log_listener = DefaultConnLogListener(self._device_log_service)
            self._client.add_connect_listener(default_conn_log_listener)

            default_conn_action_log_listener = DefaultConnActionLogListener(self._device_log_service)
            self._client.set_connect_action_listener(default_conn_action_log_listener)

        return self._client.connect()

    def get_client(self) -> DeviceClient:
        """
        获取设备客户端。获取到设备客户端后，可以直接调用客户端提供的消息、属性、命令等接口。

        Returns:
            DeviceClient: 设备客户端实例
        """
        return self._client

    def add_service(self, service_id: str, device_service: AbstractService):
        """
        添加服务。用户基于AbstractService定义自己的设备服务，并添加到设备。

        Args:
            service_id:      服务id，要和设备模型定义一致
            device_service:  服务实例
        """
        device_service.service_id = service_id
        device_service.set_iot_device(self)
        self._services[service_id] = device_service

    def get_service(self, service_id: str):
        """
        通过服务id获取服务实例

        Args:
            service_id:  服务id
        Returns:
            AbstractService: 服务实例。若服务不存在，则返回None
        """
        if service_id in self._services.keys():
            return self._services.get(service_id)
        else:
            self._logger.debug("device have no service named: %s", service_id)
            return None

    def fire_properties_changed(self, service_id: str, properties: list):
        """
        触发属性变化，SDK会上报变化的属性

        Args:
            service_id: 服务id
            properties: 属性列表
        """
        device_service: AbstractService = self.get_service(service_id)
        if device_service is None:
            self._logger.warning("device_service is None: %s", service_id)
            return
        props: dict = device_service.on_read(properties)
        service_property: ServiceProperty = ServiceProperty()
        service_property.service_id = device_service.service_id
        service_property.properties = props
        services = [service_property]
        self._client.report_properties(services)

    def fire_services_changed(self, service_ids: List[str]):
        """
        触发多个服务的属性变化,SDK自动上报变化的属性到平台

        Args:
            service_ids: 发生变化的服务id列表
        """
        services = list()
        for service_id in service_ids:
            device_service = self.get_service(service_id)
            if device_service is None:
                self._logger.warning("device_service is None: %s", service_id)
                continue
            props: dict = device_service.on_read([])
            service_property = ServiceProperty()
            service_property.service_id = device_service.service_id
            service_property.properties = props
            service_property.event_time = get_event_time()
            services.append(service_property)
        if len(services) == 0:
            return
        self.get_client().report_properties(services)

    def get_device_id(self):
        """
        获取设备id

        Returns:
            str: 设备id
        """
        return self._device_id

    def on_command(self, request_id: str, command: Command):
        """
        命令回调函数，由SDK自动调用

        Args:
            request_id:  请求id
            command:     命令
        """
        service: AbstractService = self.get_service(command.service_id)
        if service is not None:
            rsp: CommandRsp = service.on_command(command)
            self._client.respond_command(request_id, rsp)
        else:
            self._logger.warning("service is None: %s", command.service_id)

    def on_properties_set(self, request_id: str, props_set: PropSet):
        """
        属性设置回调，由SDK自动调用

        Args:
            request_id:  请求id
            props_set:   属性设置请求
        """
        for service_property in props_set.services:
            service_property: ServiceProperty
            device_service: IService = self.get_service(service_property.service_id)
            if device_service is not None:
                # 如果部分失败直接返回
                result: IotResult = device_service.on_write(service_property.properties)
                if result.result_code != SUCCESS.result_code:
                    self._client.respond_properties_set(request_id, result)
                    return
        self._client.respond_properties_set(request_id, SUCCESS)

    def on_properties_get(self, request_id: str, props_get: PropsGet):
        """
        属性查询回调，由SDK自动调用

        Args:
            request_id:  请求id
            props_get:   属性查询请求
        """
        services: List[ServiceProperty] = list()
        if props_get.service_id == "":
            # 查询所有
            for ss in iter(self._services):
                device_service: IService = self.get_service(ss)
                if device_service is not None:
                    props = device_service.on_read([])
                    service_property: ServiceProperty = ServiceProperty()
                    service_property.service_id = ss
                    service_property.properties = props
                    services.append(service_property)
        else:
            device_service: IService = self.get_service(props_get.service_id)
            if device_service is not None:
                props = device_service.on_read([])
                service_property: ServiceProperty = ServiceProperty()
                service_property.service_id = props_get.service_id
                service_property.properties = props
                services.append(service_property)
        self._client.respond_properties_get(request_id, services)

    def on_event(self, device_events: DeviceEvents):
        """
        事件回调，由SDK自动调用

        Args:
            device_events: 事件
        """
        # 子设备的
        if device_events.device_id and device_events.device_id != self.get_device_id():
            self._logger.debug("receive event for sub devices: %s", device_events.device_id)
            return

        for event in device_events.services:
            event: DeviceEvent
            if event.service_id == "$sub_device_manager":
                # 网关相关的服务，在AbstractGateway的on_event()方法中处理
                continue
            device_service: IService = self.get_service(event.service_id)
            if device_service is not None:
                # 对于系统定义的事件，sdk都有对应的服务进行处理
                device_service.on_event(event)
            else:
                self._logger.warning("device_service is None: %s", event.service_id)

    def on_rule_action_handler(self, action_list: List[Action]):
        if action_list:
            for action in action_list:
                if self.get_device_id() != action.device_id:
                    self._logger.warning("action device is not match: target: %s, action: %s", self.get_device_id(),
                                         action.device_id)
                    continue
                self.get_rule_manage_service().handle_action(action)
            return
        self._logger.warning("rule action list is empty.")

    def on_device_message(self, message: DeviceMessage):
        """
        消息回调，由SDK自动调用

        Args:
            message: 消息
        """
        # 不实现

    def get_ota_service(self) -> OTAService:
        """
        获取OTA服务

        Returns:
            OTAService: OTA服务
        """
        return self._ota_service

    def get_time_sync_service(self) -> TimeSyncService:
        """
        获取时间同步服务

        Returns:
            TimeSyncService: 时间同步服务
        """
        return self._time_sync_service

    def get_device_log_service(self) -> DeviceLogService:
        """
        获取设备日志服务

        Returns:
            DeviceLogService: 设备日志服务
        """
        return self._device_log_service

    def get_file_manager_service(self) -> FileManagerService:
        """
        获取文件管理服务

        Returns:
            FileManagerService: 文件管理服务
        """
        return self._file_manager

    def get_rule_manage_service(self) -> RuleManageService:
        return self._rule_manage_service

    def destroy(self):
        """
        释放连接
        """
        self._client.close()
