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
设备客户端，提供和平台的通讯能力，包括：

消息：双向，异步，不需要定义模型

属性：双向，设备可以上报属性，平台可以向设备读写属性，属性需要在模型定义

命令：单向，同步，平台向设备调用设备的命令

时间：双向，异步，需要在模型定义

用户不能直接创建DeviceClient实例，只能先创建IoTDevice实例，然后通过IoTDevice的get_client方法获取DeviceClient实例
"""

from __future__ import absolute_import, division, annotations
from typing import TYPE_CHECKING, List, Optional
import json
import logging
import time
import traceback
import sys
import os
import stat

from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.client.iot_result import IotResult
from iot_device_sdk_python.client.listener.command_listener import CommandListener
from iot_device_sdk_python.client.listener.device_message_listener import DeviceMessageListener
from iot_device_sdk_python.client.listener.device_shadow_listener import DeviceShadowListener
from iot_device_sdk_python.client.listener.property_listener import PropertyListener
from iot_device_sdk_python.client.listener.raw_device_message_listener import RawDeviceMessageListener
from iot_device_sdk_python.client.mqtt_connect_conf import MqttConnectConf
from iot_device_sdk_python.client.request.device_message import DeviceMessage
from iot_device_sdk_python.client.request.raw_device_message import RawDeviceMessage
from iot_device_sdk_python.client.request.shadow_data import ShadowData
from iot_device_sdk_python.transport.mqtt.mqtt_connection import MqttConnection
from iot_device_sdk_python.client.request.device_event import DeviceEvent
from iot_device_sdk_python.client.request.device_events import DeviceEvents
from iot_device_sdk_python.transport.raw_message import RawMessage
from iot_device_sdk_python.transport.raw_message_listener import RawMessageListener
from iot_device_sdk_python.transport.action_listener import ActionListener
from iot_device_sdk_python.transport.connection import Connection
from iot_device_sdk_python.utils.iot_util import get_request_id_from_msg, str_is_empty, get_event_time
from iot_device_sdk_python.client.request.service_property import ServiceProperty
from iot_device_sdk_python.client.request.command import Command
from iot_device_sdk_python.client.request.command_response import CommandRsp
from iot_device_sdk_python.client.request.props_set import PropSet
from iot_device_sdk_python.client.request.props_get import PropsGet
from iot_device_sdk_python.client.request.device_base_info import DeviceBaseInfo
from iot_device_sdk_python.rule.model.action_handler import ActionHandler
from iot_device_sdk_python.rule.model.actions import Action

if TYPE_CHECKING:
    from iot_device_sdk_python.service.abstract_device import AbstractDevice


class DeviceClient(RawMessageListener):
    _logger = logging.getLogger(__name__)
    # SDK版本信息，不能更改
    __SDK_VERSION = "Python_v1.2.0"
    __SERVER_INFO_PATH = os.path.join(sys.path[0], "server_info.json")
    __SERVER_URI = "server_uri"
    __PORT = "port"
    __SECRET = "secret"
    __BOOTSTRAP_TIMEOUT = 10.0

    def __init__(self, connect_auth_info: ConnectAuthInfo, mqtt_connect_conf: MqttConnectConf, device: AbstractDevice):
        self.check_connect_auth_info(connect_auth_info)
        self.check_mqtt_connect_conf(mqtt_connect_conf)
        self.__connect_auth_info = connect_auth_info
        self.__mqtt_connect_conf = mqtt_connect_conf

        self._device = device
        self.__connection: Optional[Connection] = None
        if self.__connect_auth_info.protocol == ConnectAuthInfo.PROTOCOL_MQTT:
            self.__connection = MqttConnection(connect_auth_info, mqtt_connect_conf, self)
        else:
            self._logger.error("Current SDK only supports PROTOCOL_MQTT.")
            return


        # 设备发放是否成功
        self.__bs_flag = False
        # 是否开启端侧规则
        self.__enable_rule_manage = connect_auth_info.enable_rule_manage
        # raw_msg_listener是原始消息接收监听器
        self.__raw_msg_listener_map = dict()
        # 设置原始消息监听器，用于接收平台下发的原始设备消息
        self.__raw_device_msg_listener: Optional[RawDeviceMessageListener] = None
        # 设置消息监听器，用于接收平台下发的设备消息
        self.__device_msg_listener: Optional[DeviceMessageListener] = None
        # 属性监听器，用于接收平台下发的属性读写操作
        self.__property_listener: Optional[PropertyListener] = None
        # 命令监听器，用于接收平台下发的命令
        self.__command_listener: Optional[CommandListener] = None
        # 影子监听器，用于接收平台下发的设备影子数据
        self.__shadow_listener: Optional[DeviceShadowListener] = None
        # 端侧规则监听器，用于自定义处理端侧规则
        self.__rule_action_handler: Optional[ActionHandler] = None

    @staticmethod
    def check_connect_auth_info(connect_auth_info: ConnectAuthInfo):
        """
        检查连接鉴权配置。若配置有问题，则抛出错误

        Args:
            connect_auth_info: 连接鉴权配置
        """
        if connect_auth_info is None:
            raise ValueError("ConnectAuthInfo is null")
        if str_is_empty(connect_auth_info.id):
            raise ValueError("ConnectAuthInfo id is invalid")
        if connect_auth_info.protocol != ConnectAuthInfo.PROTOCOL_MQTT:
            # 当前SDK只支持MQTT协议
            raise ValueError("ConnectAuthInfo protocol is invalid, currently protocol only support MQTT")
        if str_is_empty(connect_auth_info.server_uri):
            raise ValueError("ConnectAuthInfo server_uri is invalid")
        if connect_auth_info.auth_type not in [ConnectAuthInfo.SECRET_AUTH, ConnectAuthInfo.X509_AUTH]:
            raise ValueError("ConnectAuthInfo auth_type is invalid")
        if str_is_empty(connect_auth_info.secret) and (
                str_is_empty(connect_auth_info.cert_path) or str_is_empty(connect_auth_info.key_path)) is None:
            raise ValueError("ConnectAuthInfo secret or certificate is invalid")
        if connect_auth_info.port != 1883 and connect_auth_info.port != 8883:
            raise ValueError("ConnectAuthInfo port is invalid")
        if connect_auth_info.port == 8883 and str_is_empty(connect_auth_info.iot_cert_path):
            raise ValueError("ConnectAuthInfo iot_cert_path is invalid")
        if connect_auth_info.bs_mode not in [ConnectAuthInfo.BS_MODE_DIRECT_CONNECT,
                                             ConnectAuthInfo.BS_MODE_STANDARD_BOOTSTRAP,
                                             ConnectAuthInfo.BS_MODE_BOOTSTRAP_WITH_SCOPEID]:
            raise ValueError("ConnectAuthInfo bs_mode is invalid")
        if connect_auth_info.bs_mode == ConnectAuthInfo.BS_MODE_BOOTSTRAP_WITH_SCOPEID and str_is_empty(
                connect_auth_info.scope_id):
            raise ValueError("ConnectAuthInfo scope_id is invalid")
        if connect_auth_info.check_timestamp not in ["0", "1"]:
            raise ValueError("ConnectAuthInfo check_timestamp is invalid")

    @staticmethod
    def check_mqtt_connect_conf(mqtt_connect_conf: MqttConnectConf):
        """
        检查mqtt配置。若配置有问题，则抛出错误

        Args:
            mqtt_connect_conf: mqtt配置
        """
        if mqtt_connect_conf is None:
            raise ValueError("MqttConnectConf is null")
        if not isinstance(mqtt_connect_conf.keep_alive_time, int) \
                or mqtt_connect_conf.keep_alive_time < 30 \
                or mqtt_connect_conf.keep_alive_time > 1200:
            raise ValueError("MqttConnectConf keep_alive_time is invalid")
        if not isinstance(mqtt_connect_conf.qos, int) or mqtt_connect_conf.qos not in [0, 1]:
            raise ValueError("MqttConnectConf qos is invalid")
        if not isinstance(mqtt_connect_conf.timeout, float):
            raise ValueError("MqttConnectConf timeout is invalid")

    def connect(self):
        """
        和平台建立连接。连接成功时，SDK将自动向平台订阅系统定义的topic

        Returns:
            int: 结果码，0表示连接成功，其他表示连接失败
        """
        if self.__connect_auth_info.bs_mode == ConnectAuthInfo.BS_MODE_STANDARD_BOOTSTRAP or \
                self.__connect_auth_info.bs_mode == ConnectAuthInfo.BS_MODE_BOOTSTRAP_WITH_SCOPEID:
            # 设备发放场景
            if os.path.exists(self.__SERVER_INFO_PATH):
                server_info_dict = dict()
                try:
                    # 已成功进行过设备发放，从文件中读取iot平台连接信息
                    with open(self.__SERVER_INFO_PATH, 'r') as server_info:
                        server_info_dict: dict = json.load(server_info)
                except Exception:
                    self._logger.error("load server_info failed, traceback: %s", traceback.format_exc())

                if "server_uri" in server_info_dict.keys() and "port" in server_info_dict.keys():
                    server_uri = server_info_dict.get(self.__SERVER_URI)
                    port = server_info_dict.get(self.__PORT)
                    secret = server_info_dict.get(self.__SECRET)
                    self.__connect_auth_info.server_uri = server_uri
                    self.__connect_auth_info.port = port
                    if secret:
                        self.__connect_auth_info.secret = secret
                else:
                    # 进行设备发放
                    rc = self.__bootstrap()
                    if rc != 0:
                        # 发放失败
                        self._logger.error("bootstrap device failed.")
                        return rc
            else:
                # 进行设备发放
                rc = self.__bootstrap()
                if rc != 0:
                    # 发放失败
                    self._logger.error("bootstrap device failed.")
                    return rc
        # 建立设备到iot平台的连接， 将连接模式设置为直连。
        self.__connect_auth_info.bs_mode = ConnectAuthInfo.BS_MODE_DIRECT_CONNECT
        rc = self.__connect()
        if rc != 0:
            return rc
        if self.__connect_auth_info.auto_report_device_info:
            # 建链成功后，SDK自动上报版本号，软固件版本号由设备上报
            self.report_device_info(DeviceBaseInfo())
        return rc

    def __connect(self):
        """
        和平台建立连接。连接成功时，SDK将自动向平台订阅系统定义的topic

        Returns:
            int: 结果码，0表示连接成功，其他表示连接失败
        """
        return self.__connection.connect()

    def __bootstrap(self):
        """
        进行设备发放流程，返回 0表示成功，返回其它表示失败
        """
        rc = self.__connect()
        if rc != 0:
            return rc
        bs_topic = "$oc/devices/" + self.__connect_auth_info.id + "/sys/bootstrap/down"
        self.__connection.subscribe_topic(bs_topic, 0)
        topic = "$oc/devices/" + self.__connect_auth_info.id + "/sys/bootstrap/up"
        raw_message = RawMessage(topic, self.__connect_auth_info.bs_message)
        self.publish_raw_message(raw_message)
        start_time = time.time()
        while True:
            # 等待设备发放成功
            time.sleep(1)
            if self.__bs_flag:
                break
            now = time.time()
            if now - start_time > self.__BOOTSTRAP_TIMEOUT:
                self._logger.error("bootstrap failed, timeout.")
                return -1
        # 释放设备到发放服务端的连接
        self.close()
        return rc

    def _get_connection(self):
        return self.__connection

    def close(self):
        """ 释放connection连接 """
        self.__connection.close()

    def on_message_received(self, message: RawMessage):
        """
        收到原始消息后，依据topic的不同，调用不同的方法

        Args:
            message: 原始数据
        """
        try:
            topic = message.topic

            # 若订阅了自定义的topic，这里先检查接收到的topic是否是自定义的
            raw_msg_listener = self.__raw_msg_listener_map[topic] if topic in self.__raw_msg_listener_map else None
            if raw_msg_listener is not None:
                raw_msg_listener.on_message_received(message)
                return

            if "/messages/down" in topic:
                self.on_device_msg(message)  # 平台下发消息到设备测
            elif "sys/commands/request_id" in topic:
                self.on_command(message)  # 平台下发指令到设备侧
            elif "/sys/properties/set/request_id" in topic:
                self.on_properties_set(message)  # 处理写属性操作
            elif "/sys/properties/get/request_id" in topic:
                self.on_properties_get(message)  # 处理读属性操作
            elif "/sys/shadow/get/response" in topic:
                self.on_device_shadow(message)  # 处理获取平台设备影子数据
            elif "/sys/events/down" in topic:
                self.on_event(message)  # 处理平台下发事件
            elif "/sys/bootstrap/down" in topic:
                self.on_bootstrap(message)
            else:
                self._logger.warning("unknown topic: %s", topic)

        except Exception as e:
            self._logger.error("on_message_received error, tracback: %s", traceback.format_exc())
            self._logger.error("on_message_received error: %s", str(e))

    def report_device_message(self, device_message: DeviceMessage, listener: Optional[ActionListener] = None):
        """
        上报设备消息

        Args:
            device_message:  设备消息
            listener:        发布监听器，若不设置监听器则设为None
        """
        topic = '$oc/devices/' + self.__connect_auth_info.id + '/sys/messages/up'
        try:
            payload = json.dumps(device_message.to_dict())
        except Exception as e:
            self._logger.error("json.dumps failed, Exception: %s", str(e))
            raise e
        self.publish_raw_message(RawMessage(topic, payload, self.__mqtt_connect_conf.qos), listener)

    def report_properties(self, services: List[ServiceProperty], listener: Optional[ActionListener] = None):
        """
        上报设备属性

        Args:
            services:  设备属性列表
            listener:  发布监听器，若不设置监听器则设为None
        """
        topic = '$oc/devices/' + self.__connect_auth_info.id + '/sys/properties/report'
        service_list = list()
        for service in services:
            service_list.append(service.to_dict())
        try:
            payload = json.dumps({"services": service_list})
        except Exception as e:
            self._logger.error("json.dumps failed, Exception: %s", str(e))
            raise e

        self.publish_raw_message(RawMessage(topic, payload, self.__mqtt_connect_conf.qos), listener)
        """
        处理端侧规则
        """
        if self.__enable_rule_manage:
            self._device.get_rule_manage_service().handle_rule(services)

    def get_device_shadow(self, request_id: str, service_id: Optional[str] = None,
                          object_device_id: Optional[str] = None, listener: Optional[ActionListener] = None):
        """
        设备侧获取平台的设备影子数据

        Args:
            request_id:  请求id
            service_id:  服务id
            object_device_id:   device_id
            listener:    发布监听器，若不设置监听器则设为None
        """
        if object_device_id is not None:
            topic = "$oc/devices/" + object_device_id + "/sys/shadow/get/request_id=" + request_id
        else:
            topic = "$oc/devices/" + self.__connect_auth_info.id + "/sys/shadow/get/request_id=" + request_id

        payload_dict = dict()
        if service_id is not None:
            payload_dict["service_id"] = service_id
        try:
            payload = json.dumps(payload_dict)
        except Exception as e:
            self._logger.error("json.dumps failed, Exception: %s", str(e))
            raise e
        self.publish_raw_message(RawMessage(topic, payload, self.__mqtt_connect_conf.qos), listener)

    def report_event(self, device_event: DeviceEvent, listener: Optional[ActionListener] = None):
        """
        事件上报

        Args:
            device_event:    事件
            listener:        发布监听器，若不设置监听器则设为None
        """
        device_events = DeviceEvents()
        device_events.device_id = self.__connect_auth_info.id
        device_events.services = [device_event]

        topic = "$oc/devices/" + self.__connect_auth_info.id + "/sys/events/up"
        try:
            payload = json.dumps(device_events.to_dict())
        except Exception as e:
            self._logger.error("json.dumps failed, Exception: %s", str(e))
            raise e
        self.publish_raw_message(RawMessage(topic, payload, self.__mqtt_connect_conf.qos), listener)

    def report_sub_event(self, sub_device_id: str, device_event: DeviceEvent,
                         listener: Optional[ActionListener] = None):
        """
        子设备事件上报

        Args:
            sub_device_id:   子设备ID
            device_event:    事件
            listener:        发布监听器，若不设置监听器则设为None
        """
        device_events = DeviceEvents()
        device_events.device_id = self.__connect_auth_info.id
        if sub_device_id is not None:
            device_events.device_id = sub_device_id
        device_events.services = [device_event]

        topic = "$oc/devices/" + self.__connect_auth_info.id + "/sys/events/up"
        try:
            payload = json.dumps(device_events.to_dict())
        except Exception as e:
            self._logger.error("json.dumps failed, Exception: %s", str(e))
            raise e
        self.publish_raw_message(RawMessage(topic, payload, self.__mqtt_connect_conf.qos), listener)

    def respond_command(self, request_id: str, command_response: CommandRsp, listener: Optional[ActionListener] = None):
        """
        上报命令响应

        Args:
            request_id:          请求id，响应的请求id必须和请求的一致
            command_response:    命令响应
            listener:            发布监听器
        """
        topic = "$oc/devices/" + self.__connect_auth_info.id + "/sys/commands/response/request_id=" + request_id
        try:
            payload = json.dumps(command_response.to_dict())
        except Exception as e:
            self._logger.error("json.dumps failed, Exception: %s", str(e))
            raise e
        self.publish_raw_message(RawMessage(topic, payload, self.__mqtt_connect_conf.qos), listener)

    def publish_raw_message(self, raw_message: RawMessage, listener: Optional[ActionListener] = None):
        """
        发布消息

        Args:
            raw_message: 消息
            listener:    发布监听器，若不设置监听器则设为None
        """
        self.__connection.publish_message(raw_message, listener)

    def on_device_msg(self, message: RawMessage):
        """
        处理平台消息下发。若当前DeviceClient设置了消息监听器，则执行此消息监听器的on_device_message()方法。

        Args:
            message: 原始数据
        """
        self._logger.debug(f"receive message from platform, topic = %s, msg = %s", message.topic, message.payload)

        raw_device_message = RawDeviceMessage(message.payload)
        device_msg = raw_device_message.to_device_message()

        if self.__raw_device_msg_listener is not None:
            self.__raw_device_msg_listener.on_raw_device_message(raw_device_message)

        if device_msg is not None:
            is_current_device = device_msg.device_id is None \
                                or len(device_msg.device_id) == 0 \
                                or device_msg.device_id == self.__connect_auth_info.id
            if self.__device_msg_listener is not None and is_current_device:
                self.__device_msg_listener.on_device_message(device_msg)
            else:
                self._device.on_device_message(device_msg)

    def on_command(self, message: RawMessage):
        """
        处理平台命令下发。若当前DeviceClient设置了命令监听器，则执行此命令监听器的on_command()方法。

        Args:
            message: 原始数据
        """
        request_id = get_request_id_from_msg(message)
        try:
            self._logger.debug("receive command from platform, topic = %s, msg = %s", message.topic,
                               str(message.payload))
            cmd = json.loads(message.payload)
        except Exception as e:
            self._logger.error("json.loads failed, Exception: %s", str(e))
            raise e
        command = Command()
        command.convert_from_dict(cmd)
        if self.__command_listener is not None:
            self.__command_listener.on_command(request_id, command.service_id, command.command_name,
                                               command.paras)
        else:
            self._device.on_command(request_id, command)

    def on_rule_command(self, request_id: str, command: Command):
        if self.__command_listener is not None:
            self.__command_listener.on_command(request_id, command.service_id, command.command_name, command.paras)
            return
        self._logger.warning("command listener was not config for rules.")

    def on_device_shadow(self, message: RawMessage):
        """
        处理平台设备影子数据下发。若当前DeviceClient设置了影子监听器，则执行此命令监听器的on_shadow_get()方法。

        Args:
            message: 原始数据
        """
        request_id = get_request_id_from_msg(message)
        try:
            payload: dict = json.loads(message.payload)
            device_id: str = payload.get("object_device_id")
            shadow_list: List[ShadowData] = list()
            shadow_dict_list: list = payload.get("shadow")
            for shadow_dict in shadow_dict_list:
                shadow = ShadowData()
                shadow.convert_from_dict(shadow_dict)
                shadow_list.append(shadow)
            if self.__shadow_listener is not None:
                self.__shadow_listener.on_shadow_get(request_id, device_id, shadow_list)
            else:
                # 没有设置影子监听器，这里不做处理
                pass
        except Exception as e:
            self._logger.error("handle device shadow failed, Exception: %s", str(e))
            pass

    def on_properties_set(self, message: RawMessage):
        """
        处理平台设置设备属性。若当前DeviceClient设置了属性监听器，则执行此命令监听器的on_property_set()方法。

        Args:
            message: 原始数据
        """
        request_id = get_request_id_from_msg(message)
        try:
            self._logger.debug("receive properties_set from platform, topic = %s, msg = %s",
                               message.topic, str(message.payload))
            payload: dict = json.loads(message.payload)
        except Exception as e:
            self._logger.error("json.loads failed, Exception: %s", str(e))
            raise e
        prop_set: PropSet = PropSet()
        service_list: list = payload["services"]
        service_property_list = [ServiceProperty(service_id=a["service_id"],
                                                 properties=a["properties"]) for a in service_list]
        prop_set.services = service_property_list
        if self.__property_listener is not None:
            self.__property_listener.on_property_set(request_id, prop_set.services)
        else:
            self._device.on_properties_set(request_id, prop_set)

    def on_properties_get(self, message: RawMessage):
        """
        处理平台查询设备属性。若当前DeviceClient设置了属性监听器，则执行此命令监听器的on_property_get()方法。

        Args:
            message: 原始数据
        """
        request_id = get_request_id_from_msg(message)
        try:
            self._logger.debug("receive properties_get from platform, topic = %s, msg = %s", message.topic,
                               str(message.payload))
            obj = json.loads(message.payload)
        except Exception as e:
            self._logger.error("json.loads failed, Exception: %s", str(e))
            raise e
        prop_get = PropsGet()
        prop_get.convert_from_dict(obj)
        if self.__property_listener is not None:
            self.__property_listener.on_property_get(request_id, prop_get.service_id)
        else:
            self._device.on_properties_get(request_id, prop_get)

    def on_event(self, message: RawMessage):
        """
        处理平台事件下发

        Args:
            message: 原始数据
        """
        try:
            self._logger.debug("receive events from platform, topic = %s, msg = %s", message.topic,
                               str(message.payload))
            payload: dict = json.loads(message.payload)
        except Exception as e:
            self._logger.error("json.loads failed, Exception: %s", str(e))
            raise e
        device_events = DeviceEvents()
        device_events.convert_from_dict(payload)
        if not device_events:
            self._logger.error("device events invalid, payload: %s", str(payload))
            return
        self._device.on_event(device_events)

    def on_rule_action_handler(self, action: List[Action]):
        """
        处理端侧规则

        Args:
            action: 原始数据
        """
        if self.__rule_action_handler is not None:
            self.__rule_action_handler.handle_rule_action(action)
            return
        self._device.on_rule_action_handler(action)

    def on_bootstrap(self, message: RawMessage):
        """
        处理设备发放信息

        Args:
            message: 原始数据
        """
        try:
            self._logger.debug("receive bootstrap info from platform, topic = %s, msg = %s", message.topic,
                               str(message.payload))
            payload: dict = json.loads(message.payload)
        except Exception as e:
            self._logger.error("json.loads failed, Exception: %s", str(e))
            raise e
        address = str(payload.get("address"))
        device_secret = payload.get("deviceSecret")
        self.__connect_auth_info.server_uri = address.split(":")[0]
        self.__connect_auth_info.port = int(address.split(":")[-1])
        if device_secret:
            self.__connect_auth_info.secret = device_secret
        # 设备发放成功，保存获取的iot平台地址和端口
        if os.path.exists(self.__SERVER_INFO_PATH):
            os.remove(self.__SERVER_INFO_PATH)
        server_info_dict = {self.__SERVER_URI: self.__connect_auth_info.server_uri,
                            self.__PORT: self.__connect_auth_info.port,
                            self.__SECRET: device_secret}
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(self.__SERVER_INFO_PATH, flags, modes), 'w') as server_info:
            json.dump(server_info_dict, server_info)
        self._logger.info("bootstrap success, change server address to %s", address)
        self.__bs_flag = True

    def respond_properties_get(self, request_id: str, services: List[ServiceProperty],
                               listener: Optional[ActionListener] = None):
        """
        上报读属性响应

        Args:
            request_id:  请求id，响应的请求id必须和请求的一致
            services:    设备属性列表
            listener:    发布监听器
        """
        topic = "$oc/devices/" + self.__connect_auth_info.id + "/sys/properties/get/response/request_id=" + request_id
        service_list = list()
        for service in services:
            service_list.append(service.to_dict())
        try:
            payload = json.dumps({"services": service_list})
        except Exception as e:
            self._logger.error("json.dumps failed, Exception: %s", str(e))
            raise e
        self.publish_raw_message(RawMessage(topic, payload, self.__mqtt_connect_conf.qos), listener)

    def respond_properties_set(self, request_id: str, iot_result: IotResult, listener: Optional[ActionListener] = None):
        """
        上报写属性响应

        Args:
            request_id:  请求id，响应的请求id必须和请求的一致
            iot_result:  写属性结果
            listener:    发布监听器
        """
        topic = "$oc/devices/" + self.__connect_auth_info.id + "/sys/properties/set/response/request_id=" + request_id
        try:
            payload = json.dumps(iot_result.to_dict())
        except Exception as e:
            self._logger.error("json.dumps failed, Exception: %s", str(e))
            raise e
        self.publish_raw_message(RawMessage(topic, payload, self.__mqtt_connect_conf.qos), listener)

    def set_raw_device_msg_listener(self, raw_device_msg_listener: RawDeviceMessageListener):
        """
        设置原始消息监听器，用于接收平台下发的消息，消息保持为二进制格式。
        需要通过IoTDevice的getClient方法获取DeviceClient实例后，调用此方法设置消息监听器

        Args:
            raw_device_msg_listener:     消息监听器
        """
        if not isinstance(raw_device_msg_listener, RawDeviceMessageListener):
            self._logger.error("device_msg_listener should be RawDeviceMessageListener type")
            return
        self.__raw_device_msg_listener = raw_device_msg_listener

    def set_device_msg_listener(self, device_msg_listener: DeviceMessageListener):
        """
        设置消息监听器，用于接收平台下发的消息。
        需要通过IoTDevice的getClient方法获取DeviceClient实例后，调用此方法设置消息监听器

        Args:
            device_msg_listener:     消息监听器
        """
        if not isinstance(device_msg_listener, DeviceMessageListener):
            self._logger.error("device_msg_listener should be DeviceMessageListener type")
            return
        self.__device_msg_listener = device_msg_listener

    def set_properties_listener(self, property_listener: PropertyListener):
        """
        设置属性监听器，用于接收平台下发的属性读写
        需要通过IoTDevice的getClient方法获取DeviceClient实例后，调用此方法设置消息监听器

        Args:
            property_listener:   属性监听器
        """
        if not isinstance(property_listener, PropertyListener):
            self._logger.error("property_listener should be PropertyListener")
            return
        self.__property_listener = property_listener

    def set_command_listener(self, command_listener: CommandListener):
        """
        设置命令监听器，用于接收平台下发的命令。
        需要通过IoTDevice的getClient接口获取DeviceClient实例后，调用此方法设置命令监听器

        Args:
            command_listener:    命令监听器
        """
        if not isinstance(command_listener, CommandListener):
            self._logger.error("command_listener should be CommandListener")
            return
        self.__command_listener = command_listener

    def set_device_shadow_listener(self, device_shadow_listener: DeviceShadowListener):
        """
        设置影子监听器，用于接收平台下发的设备影子数据
        需要通过IoTDevice的getClient方法获取DeviceClient实例后，调用此方法设置消息监听器

        Args:
            device_shadow_listener: 影子监听器
        """
        if not isinstance(device_shadow_listener, DeviceShadowListener):
            self._logger.error("device_shadow_listener should be DeviceShadowListener")
            return
        self.__shadow_listener = device_shadow_listener

    def set_rule_action_handler(self, rule_action_handler: ActionHandler):
        """
        设置端侧规则监听器，用于自定义处理端侧规则
        需要通过IoTDevice的getClient方法获取DeviceClient实例后，调用此方法设置消息监听器

        Args:
            rule_action_handler: 端侧规则监听器
        """
        if not isinstance(rule_action_handler, ActionHandler):
            self._logger.error("rule_action_handler should be ActionHandler")
            return
        self.__rule_action_handler = rule_action_handler

    def subscribe_topic(self, topic: str, qos: int, message_listener):
        """
        订阅自定义topic。此接口只能用于订阅自定义topic
        需要通过IoTDevice的getClient方法获取DeviceClient实例后，调用此方法设置消息监听器

        Args:
            topic:               自定义topic
            qos:                 qos
            message_listener:    接收自定义消息的监听器
        """
        self.__connection.subscribe_topic(topic, qos)
        self.__raw_msg_listener_map[topic] = message_listener

    def add_connect_listener(self, connect_listener):
        """
        设置链路监听器，用户接收链路建立和断开事件

        Args:
            connect_listener: 链路监听器
        """
        self.__connection.add_connect_listener(connect_listener)

    def set_connect_action_listener(self, connect_action_listener):
        """
        设置连接动作监听器，用户接受连接成功或失败的事件

        Args:
            connect_action_listener: 连接动作监听器
        """
        self.__connection.set_connect_action_listener(connect_action_listener)

    def report_device_info(self, device_info: DeviceBaseInfo, listener: Optional[ActionListener] = None):
        """
        上报设备信息，包括：软件版本，硬件版本以及SDK版本
        需要通过IoTDevice的getClient方法获取DeviceClient实例后，调用此方法设置消息监听器

        Args:
            device_info:  设备信息
            listener:    发布监听器，若不设置监听器则设为None
        """
        device_event = DeviceEvent()
        device_event.service_id = "$sdk_info"
        device_event.event_type = "sdk_info_report"
        device_event.event_time = get_event_time()
        paras: dict = {"device_sdk_version": self.__SDK_VERSION,
                       "sw_version": device_info.sw_version,
                       "fw_version": device_info.fw_version}
        device_event.paras = paras
        self.report_event(device_event, listener)

    def enable_rule_manage(self):
        return self.__enable_rule_manage
