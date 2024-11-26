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

import asyncio
import logging
import traceback

from tornado.iostream import IOStream

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.request.command import Command
from iot_device_sdk_python.client.request.command_response import CommandRsp
from iot_device_sdk_python.client.request.device_event import DeviceEvent
from iot_device_sdk_python.client.request.device_events import DeviceEvents
from iot_device_sdk_python.client.request.device_message import DeviceMessage
from iot_device_sdk_python.client.request.service_property import ServiceProperty
from iot_device_sdk_python.gateway.abstract_gateway import AbstractGateway
from iot_device_sdk_python.gateway.requests.device_info import DeviceInfo
from iot_device_sdk_python.gateway.sub_devices_persistence import SubDevicesPersistence
from iot_gateway_demo.session import Session
from iot_device_sdk_python.utils.iot_util import get_node_id_from_device_id
from iot_device_sdk_python.client.iot_result import SUCCESS
from iot_device_sdk_python.client.request.props_set import PropSet
from iot_device_sdk_python.client.request.props_get import PropsGet


class SimpleGateway(AbstractGateway):
    """
    此例子用来演示如何使用云网关来实现TCP协议设备接入。网关和平台只建立一个MQTT连接，使用网关的身份和平台进行通讯。
    本例子TCP server传输简单的字符串，并且首条消息会发送设备标识来鉴权。
    用户可以自行扩展StringTcpServer类来实现更复杂的TCP server。
    """
    def __init__(self, sub_devices_persistence: SubDevicesPersistence, client_conf: ClientConf):
        super(SimpleGateway, self).__init__(sub_devices_persistence, client_conf)
        self._logger = logging.getLogger(__name__)
        self._node_id2session_dict = dict()

    def create_session(self, node_id: str, stream: IOStream):
        # 北向已经添加了此设备
        subdev: DeviceInfo = self.get_sub_device_by_node_id(node_id)
        if subdev is not None:
            session = Session()
            session.stream = stream
            session.node_id = node_id
            session.device_id = subdev.device_id

            self._node_id2session_dict[node_id] = session
            self._logger.info("create new session ok, node_id: " + session.node_id)
            return session
        self._logger.info("not allowed:" + node_id)
        return None

    def get_session(self, node_id: str):
        if node_id in self._node_id2session_dict.keys():
            return self._node_id2session_dict.get(node_id)
        else:
            self._logger.info("node not found in node_id2session_dict: " + node_id)
            return None

    def on_sub_dev_message(self, message: DeviceMessage):
        """
        子设备消息下发，网关需要转发给子设备
        :param message: 设备消息
        """
        if not message.device_id:
            # 对于子设备，需要指定设备id
            self._logger.error("message.device_id is None")
            return
        node_id = get_node_id_from_device_id(message.device_id)
        if not node_id:
            self._logger.error("node_id get from device_id is None")
            return
        if node_id not in self._node_id2session_dict.keys():
            self._logger.error("node_id not found in node_id2session_dict: " + node_id)
            return
        session: Session = self._node_id2session_dict.get(node_id)
        if not session:
            self._logger.error("session is None, node_id:" + node_id)
            return
        msg_content: str = str(message.content)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 将命令转发给子设备
            loop.run_until_complete(session.stream.write(bytes(msg_content.encode("utf-8"))))
        except Exception as e:
            self._logger.error("session.stream.write failed, Exception: " + traceback.format_exc())
        finally:
            loop.close()
        self._logger.info("gateway sent device message to sub device")

    def on_sub_dev_command(self, request_id: str, command: Command):
        """
        子设备命令下发处理，网关需要转发给子设备
        :param request_id:  请求id
        :param command:     命令
        """
        if not command.device_id:
            # 对于子设备，需要指定设备id
            self._logger.error("command.device_id is None")
            return
        node_id = get_node_id_from_device_id(command.device_id)
        if not node_id:
            self._logger.error("node_id get from device_id is None")
            return
        if node_id not in self._node_id2session_dict.keys():
            self._logger.error("node_id not found in node_id2session_dict: " + node_id)
            return
        session: Session = self._node_id2session_dict.get(node_id)
        if not session:
            self._logger.error("session is None, node_id:" + node_id)
            return
        # 这里直接把paras字典转成string发给子设备
        str_paras = str(command.paras)
        self._logger.info("command paras is: " + str_paras)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 将命令转发给子设备
            loop.run_until_complete(session.stream.write(bytes(str_paras.encode("utf-8"))))
        except Exception as e:
            self._logger.error("session.stream.write failed, Exception: " + traceback.format_exc())
        finally:
            loop.close()
        self._logger.info("gateway sent command to sub device")

        # 为了简化处理，我们在这里直接回命令响应。更合理的做法是在子设备处理完后再回响应
        command_rsp = CommandRsp()
        command_rsp.result_code = CommandRsp.success_code()
        command_rsp.paras = {"result": command.device_id}
        self.get_client().respond_command(request_id, command_rsp)
        self._logger.info("gateway respond_command")

    def on_sub_dev_properties_set(self, request_id: str, props_set: PropSet):
        """
        子设备属性设置处理，网关需要转发给子设备
        :param request_id:  请求id
        :param props_set:   属性设置
        """
        if not props_set.device_id:
            # 对于子设备，需要指定设备id
            self._logger.error("command.device_id is None")
            return
        node_id = get_node_id_from_device_id(props_set.device_id)
        if not node_id:
            self._logger.error("node_id get from device_id is None")
            return
        if node_id not in self._node_id2session_dict.keys():
            self._logger.error("node_id not found in node_id2session_dict: " + node_id)
            return
        session: Session = self._node_id2session_dict.get(node_id)
        if not session:
            self._logger.error("session is None, node_id:" + node_id)
            return
        # 这里直接把属性列表转成string发给子设备
        props_list = list()
        for service_property in props_set.services:
            service_property: ServiceProperty
            props_list.append(service_property.to_dict())
        str_props = str(props_list)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 将命令转发给子设备
            loop.run_until_complete(session.stream.write(bytes(str_props.encode("utf-8"))))
        except Exception as e:
            self._logger.error("session.stream.write failed, Exception: " + traceback.format_exc())
        finally:
            loop.close()
        self._logger.info("gateway sent property set to sub device")

        # 为了简化处理，我们在这里直接回属性设置响应。更合理的做法是在子设备处理完后再回响应
        self.get_client().respond_properties_set(request_id, SUCCESS)
        self._logger.info("gateway respond_properties_set")

    def on_sub_dev_properties_get(self, request_id: str, props_get: PropsGet):
        """
        子设备属性读取处理，网关需要转发给子设备。
        :param request_id:  请求id
        :param props_get:   属性读取
        """
        # 不建议平台直接读子设备的属性
        self._logger.error("not support on_sub_dev_properties_get")

    def on_sub_dev_event(self, device_events: DeviceEvents):
        node_id = get_node_id_from_device_id(device_events.device_id)
        if not node_id:
            self._logger.error("node_id get from device_id is None")
            return
        if node_id not in self._node_id2session_dict.keys():
            self._logger.error("node_id not found in node_id2session_dict: " + node_id)
            return
        session: Session = self._node_id2session_dict.get(node_id)
        if not session:
            self._logger.error("session is None, node_id:" + node_id)
            return
        # 这里直接把属性列表转成string发给子设备
        str_events = str(device_events.to_dict())

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 将命令转发给子设备
            loop.run_until_complete(session.stream.write(bytes(str_events.encode("utf-8"))))
        except Exception as e:
            self._logger.error("session.stream.write failed, Exception: " + traceback.format_exc())
        finally:
            loop.close()
        self._logger.info("gateway sent property set to sub device")

        # 为了简化处理，我们在这里直接回事件响应。更合理的做法是在子设备处理完后再回响应
        ota_type_list = ["firmware_upgrade", "firmware_upgrade", "firmware_upgrade_v2", "software_upgrade_v2"]
        for service in device_events.services:
            if service.service_id != "$ota":
                continue
            device_event = DeviceEvent()
            if service.event_type == "version_query":
                device_event.service_id = "$ota"
                device_event.event_type = "version_report"
                paras = dict()
                paras["fw_version"] = "v1.0"
                paras["sw_version"] = "v1.0"
                device_event.paras = paras
                self.get_client().report_sub_event(device_events.device_id, device_event)
            if service.event_type in ota_type_list:
                device_event.service_id = "$ota"
                device_event.event_type = "upgrade_progress_report"
                paras = dict()
                paras["result_code"] = 0
                paras["version"] = service.paras["version"]
                device_event.paras = paras
                self.get_client().report_sub_event(device_events.device_id, device_event)
        self._logger.info("gateway respond_properties_set")

