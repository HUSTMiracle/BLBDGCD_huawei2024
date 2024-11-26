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
import json

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.request.props_get import PropsGet
from iot_device_sdk_python.client.request.props_set import PropSet
from iot_device_sdk_python.iot_device import IotDevice
from iot_device_sdk_python.client.request.device_event import DeviceEvent
from iot_device_sdk_python.gateway.sub_dev_discovery_listener import SubDevDiscoveryListener
from iot_device_sdk_python.gateway.gtw_operate_sub_device_listener import GtwOperateSubDeviceListener
from iot_device_sdk_python.gateway.sub_devices_persistence import SubDevicesPersistence
from iot_device_sdk_python.transport.action_listener import ActionListener
from iot_device_sdk_python.utils.iot_util import get_node_id_from_device_id, get_event_time
from iot_device_sdk_python.gateway.requests.device_info import DeviceInfo
from iot_device_sdk_python.client.request.device_message import DeviceMessage
from iot_device_sdk_python.client.request.service_property import ServiceProperty
from iot_device_sdk_python.gateway.requests.device_property import DeviceProperty
from iot_device_sdk_python.transport.raw_message import RawMessage
from iot_device_sdk_python.gateway.requests.device_status import DeviceStatus
from iot_device_sdk_python.gateway.requests.added_sub_device_info import AddedSubDeviceInfo
from iot_device_sdk_python.client.request.device_events import DeviceEvents
from iot_device_sdk_python.gateway.requests.sub_devices_info import SubDevicesInfo
from iot_device_sdk_python.gateway.requests.gtw_add_sub_device_rsp import GtwAddSubDeviceRsp
from iot_device_sdk_python.gateway.requests.gtw_del_sub_device_rsp import GtwDelSubDeviceRsp
from iot_device_sdk_python.gateway.requests.added_sub_device_info_rsp import AddedSubDeviceInfoRsp
from iot_device_sdk_python.gateway.requests.add_sub_device_failed_reason import AddSubDeviceFailedReason
from iot_device_sdk_python.gateway.requests.del_sub_device_failed_reason import DelSubDeviceFailedReason
from iot_device_sdk_python.client.request.command import Command


class AbstractGateway(IotDevice):
    """
    抽象网关，实现了子设备管理，子设备消息转发功能
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, sub_devices_persistence: SubDevicesPersistence, client_conf: ClientConf):
        """
        初始化方法

        Args:
            sub_devices_persistence: 子设备信息持久化，提供子设备信息保存能力
        """
        super().__init__(client_conf)
        self._sub_dev_discovery_listener: Optional[SubDevDiscoveryListener] = None  # TODO 子设备发现监听器，属于scan事件类型（暂未使用）
        self._gtw_operate_sub_device_listener: Optional[GtwOperateSubDeviceListener] = None  # 网关操作子设备监听器
        self._sub_device_persistence = sub_devices_persistence  # 子设备持久化，提供子设备信息保存能力

    def set_sub_dev_discovery_listener(self, sub_dev_discovery_listener: SubDevDiscoveryListener):
        """
        设置子设备发现监听器（暂未使用）
        TODO 子设备发现监听器，属于scan事件类型，暂未使用

        Args:
            sub_dev_discovery_listener:  子设备发现监听器
        """
        self._sub_dev_discovery_listener = sub_dev_discovery_listener

    def set_gtw_operate_sub_device_listener(self, gtw_operate_sub_device_listener: GtwOperateSubDeviceListener):
        """
        设置网关添加/删除子设备监听器

        Args:
            gtw_operate_sub_device_listener: 网关操作子设备监听器
        """
        self._gtw_operate_sub_device_listener = gtw_operate_sub_device_listener

    def get_sub_device_by_node_id(self, node_id: str) -> DeviceInfo:
        """
        根据设备标识码查询子设备

        Args:
            node_id: 设备标识码

        Returns:
            DeviceInfo: 子设备信息
        """
        return self._sub_device_persistence.get_sub_device(node_id)

    def get_sub_device_by_device_id(self, device_id: str) -> DeviceInfo:
        """
        根据设备id查询子设备

        Args:
            device_id:   设备id

        Returns:
            DeviceInfo: 子设备信息，
        """
        node_id: str = get_node_id_from_device_id(device_id)
        return self._sub_device_persistence.get_sub_device(node_id)

    def report_sub_dev_list(self, device_infos: List[DeviceInfo], listener: Optional[ActionListener] = None):
        """
        上报子设备发现结果
        TODO 属于scan事件类型，暂未使用

        Args:
            device_infos:    子设备信息列表
            listener:        发布监听器
        """
        device_event = DeviceEvent()
        device_event.service_id = "sub_device_discovery"
        device_event.event_type = "scan_result"
        device_event.event_time = get_event_time()

        device_info_list = list()
        for device_info in device_infos:
            device_info_list.append(device_info.to_dict())
        paras: dict = {"devices": device_info_list}

        device_event.paras = paras
        self.get_client().report_event(device_event, listener)

    def report_sub_device_message(self, device_message: DeviceMessage, listener: Optional[ActionListener] = None):
        """
        上报子设备消息

        Args:
            device_message:  设备消息
            listener:        发布监听器
        """
        self.get_client().report_device_message(device_message, listener)

    def report_sub_device_properties(self, device_id: str, services: List[ServiceProperty],
                                     listener: Optional[ActionListener] = None):
        """
        上报子设备属性

        Args:
            device_id:   子设备id
            services:    服务属性列表
            listener:    发布监听器
        """
        device_property = DeviceProperty()
        device_property.device_id = device_id
        device_property.services = services
        self.report_batch_properties([device_property], listener)

    def report_batch_properties(self, device_properties: List[DeviceProperty],
                                listener: Optional[ActionListener] = None):
        """
        批量上报子设备属性

        Args:
            device_properties:   子设备属性列表
            listener:            发布监听器
        """
        device_property_list = list()
        for device_property in device_properties:
            device_property_list.append(device_property.to_dict())
        devices: dict = {"devices": device_property_list}
        topic = "$oc/devices/" + self.get_device_id() + "/sys/gateway/sub_devices/properties/report"
        try:
            payload = json.dumps(devices)
        except Exception as e:
            self._logger.error("json.dumps failed, Exception: %s", str(e))
            raise e
        raw_message = RawMessage(topic, payload)
        self.get_client().publish_raw_message(raw_message, listener)

    def report_sub_device_status(self, device_id: str, status: str, listener: Optional[ActionListener] = None):
        """
        上报子设备状态

        Args:
            device_id:   子设备id
            status:      设备状态。OFFLINE:设备离线；ONLINE:设备在线
            listener:    发布监听器
        """
        device_status = DeviceStatus()
        device_status.device_id = device_id
        device_status.status = status
        self.report_batch_status([device_status], listener)

    def report_batch_status(self, statuses: List[DeviceStatus], listener: Optional[ActionListener] = None):
        """
        批量上报子设备状态

        Args:
            statuses:    子设备状态列表
            listener:    发布监听器
        """
        device_event = DeviceEvent()
        device_event.service_id = "$sub_device_manager"
        device_event.event_type = "sub_device_update_status"
        device_event.event_time = get_event_time()
        status_list = list()
        for status in statuses:
            status_list.append(status.to_dict())
        device_event.paras = {"device_statuses": status_list}
        self.get_client().report_event(device_event, listener)

    def gtw_add_sub_device(self, added_sub_device_infos: List[AddedSubDeviceInfo], event_id: str,
                           listener: Optional[ActionListener] = None):
        """
        网关发起新增子设备请求

        Args:
            added_sub_device_infos:  子设备信息列表
            event_id:        此次请求的事件id，不携带则由平台自动生成
            listener:        发布监听器
        """
        device_event = DeviceEvent()
        device_event.service_id = "$sub_device_manager"
        device_event.event_type = "add_sub_device_request"
        device_event.event_time = get_event_time()
        device_event.event_id = event_id
        added_sub_device_info_list = list()
        for added_sub_device_info in added_sub_device_infos:
            added_sub_device_info_list.append(added_sub_device_info.to_dict())
        paras: dict = {"devices": added_sub_device_info_list}
        device_event.paras = paras
        self.get_client().report_event(device_event, listener)

    def gtw_del_sub_device(self, del_sub_devices: List[str], event_id: str, listener: Optional[ActionListener] = None):
        """
        网关发起删除子设备请求

        Args:
            del_sub_devices: 要删除的子设备列表
            event_id:        此次请求的事件id，不携带则有平台自动生成
            listener:        发布监听器
        """
        device_event = DeviceEvent()
        device_event.service_id = "$sub_device_manager"
        device_event.event_type = "delete_sub_device_request"
        device_event.event_time = get_event_time()
        device_event.event_id = event_id
        paras: dict = {"devices": del_sub_devices}
        device_event.paras = paras
        self.get_client().report_event(device_event, listener)

    def on_event(self, device_events: DeviceEvents):
        """
        事件处理回调，由SDK自动调用

        Args:
            device_events:   设备事件
        """
        # 子设备的事件
        if device_events.device_id and device_events.device_id != self.get_device_id():
            self.on_sub_dev_event(device_events)
            return
        # 网关的事件
        super().on_event(device_events)
        # 网关管理子设备的事件
        for event in device_events.services:
            event: DeviceEvent
            if event.service_id != "$sub_device_manager":
                continue

            if event.event_type == "start_scan":
                # TODO scan的事件类型暂未启用
                pass
            elif event.event_type == "add_sub_device_notify":
                # 平台通知网关子设备新增
                sub_devices_info = SubDevicesInfo()
                version = event.paras.get("version")
                sub_devices_info.version = version

                tmp = list()
                devices: list = event.paras.get("devices")
                for device in devices:
                    device: dict
                    device_info = DeviceInfo()
                    device_info.convert_from_dict(device)
                    tmp.append(device_info)

                sub_devices_info.devices = tmp
                self.on_add_sub_devices(sub_devices_info)
            elif event.event_type == "delete_sub_device_notify":
                # 平台通知网关子设备删除
                sub_devices_info = SubDevicesInfo()
                version = event.paras.get("version")
                sub_devices_info.version = version

                tmp = list()
                devices: list = event.paras.get("devices")
                for device in devices:
                    device: dict
                    device_info = DeviceInfo()
                    device_info.convert_from_dict(device)
                    tmp.append(device_info)

                sub_devices_info.devices = tmp
                self.on_delete_sub_devices(sub_devices_info)
            elif event.event_type == "add_sub_device_response":
                # 网关新增子设备请求响应
                gtw_add_sub_device_rsp = GtwAddSubDeviceRsp()

                # successful_devices
                success_tmp = list()
                successful_devices: list = event.paras.get("successful_devices")
                for device in successful_devices:
                    device: dict
                    added_sub_device_info_rsp = AddedSubDeviceInfoRsp()
                    added_sub_device_info_rsp.convert_from_dict(device)
                    success_tmp.append(added_sub_device_info_rsp)

                gtw_add_sub_device_rsp.successful_devices = success_tmp

                # failed_devices
                fail_tmp = list()
                failed_devices: list = event.paras.get("failed_devices")
                for device in failed_devices:
                    device: dict
                    add_sub_device_failed_reason = AddSubDeviceFailedReason()
                    add_sub_device_failed_reason.convert_from_dict(device)
                    fail_tmp.append(add_sub_device_failed_reason)

                gtw_add_sub_device_rsp.add_sub_device_failed_reasons = fail_tmp

                if self._gtw_operate_sub_device_listener is not None:
                    self._gtw_operate_sub_device_listener.on_add_sub_device_rsp(gtw_add_sub_device_rsp,
                                                                                event.event_id)
            elif event.event_type == "delete_sub_device_response":
                # 网关删除子设备请求响应
                gtw_del_sub_device_rsp = GtwDelSubDeviceRsp()

                # successful_devices
                gtw_del_sub_device_rsp.successful_devices = event.paras.get("successful_devices")

                # failed_devices
                fail_tmp = list()
                failed_devices: list = event.paras.get("failed_devices")
                for device in failed_devices:
                    device: dict
                    del_sub_device_failed_reason = DelSubDeviceFailedReason()
                    del_sub_device_failed_reason.convert_from_dict(device)
                    fail_tmp.append(del_sub_device_failed_reason)

                gtw_del_sub_device_rsp.failed_devices = fail_tmp
                if self._gtw_operate_sub_device_listener is not None:
                    self._gtw_operate_sub_device_listener.on_del_sub_device_rsp(gtw_del_sub_device_rsp,
                                                                                event.event_id)
            else:
                self._logger.info("gateway receive unknown event_type: %s", event.event_type)

    def on_device_message(self, message: DeviceMessage):
        """
        设备消息处理回调

        Args:
            message: 消息
        """
        # 子设备的
        if message.device_id and message.device_id != self.get_device_id():
            self.on_sub_dev_message(message)
            return
        # 网关的
        super().on_device_message(message)

    def on_command(self, request_id: str, command: Command):
        """
        命令处理回调

        Args:
            request_id:  请求id
            command:     命令
        """
        # 子设备的
        if command.device_id and command.device_id != self.get_device_id():
            self.on_sub_dev_command(request_id, command)
            return
        # 网关的
        super().on_command(request_id, command)

    def on_properties_set(self, request_id: str, props_set: PropSet):
        """
        属性设置处理回调

        Args:
            request_id:  请求id
            props_set:   属性设置请求
        """
        # 子设备的
        if props_set.device_id and props_set.device_id != self.get_device_id():
            self.on_sub_dev_properties_set(request_id, props_set)
            return
        # 网关的
        super().on_properties_set(request_id, props_set)

    def on_properties_get(self, request_id: str, props_get: PropsGet):
        """
        属性查询处理回调

        Args:
            request_id:  请求id
            props_get:   属性查询请求
        """
        # 子设备的
        if props_get.device_id and props_get.device_id != self.get_device_id():
            self.on_sub_dev_properties_get(request_id, props_get)
            return
        # 网关的
        super().on_properties_get(request_id, props_get)

    def on_add_sub_devices(self, sub_devices_info: SubDevicesInfo):
        """
        添加子设备处理回调

        Args:
            sub_devices_info:    子设备信息

        Returns:
            int: 处理结果，0表示成功
        """
        if self._sub_device_persistence is not None:
            return self._sub_device_persistence.add_sub_devices(sub_devices_info)
        return -1

    def on_delete_sub_devices(self, sub_devices_info: SubDevicesInfo):
        """
        删除子设备处理回调

        Args:
            sub_devices_info:    子设备信息

        Returns:
            int: 处理结果，0表示成功
        """
        if self._sub_device_persistence is not None:
            return self._sub_device_persistence.delete_sub_devices(sub_devices_info)
        return -1

    def sync_sub_devices(self):
        """
        向平台请求同步子设备信息
        """
        self._logger.debug("start to syncSubDevices, local version is %s",
                           str(self._sub_device_persistence.get_version()))

        device_event = DeviceEvent()
        device_event.service_id = "$sub_device_manager"
        device_event.event_type = "sub_device_sync_request"
        device_event.event_time = get_event_time()
        paras: dict = {"version": self._sub_device_persistence.get_version()}
        device_event.paras = paras

        self.get_client().report_event(device_event)

    def on_sub_dev_command(self, request_id: str, command: Command):
        """
        子设备命令下发处理，网关需要转发给子设备，需要子类实现

        Args:
            request_id:  请求id
            command:     命令
        """

    def on_sub_dev_properties_set(self, request_id: str, props_set: PropSet):
        """
        子设备属性设置，网关需要转发给子设备，需要子类实现

        Args:
            request_id:  请求id
            props_set:   属性设置
        """

    def on_sub_dev_properties_get(self, request_id: str, props_get: PropsGet):
        """
        子设备属性查询，网关需要转发给子设备，需要子类实现

        Args:
            request_id:  请求id
            props_get:   属性查询
        """

    def on_sub_dev_message(self, message: DeviceMessage):
        """
        子设备消息下发，网关需要转发给子设备，需要子类实现

        Args:
            message: 设备消息
        """

    def on_sub_dev_event(self, device_events: DeviceEvents):
        """
        子设备事件下发，网关需要转发给子设备，需要子类实现

        Args:
            device_events: 设备事件
        """
