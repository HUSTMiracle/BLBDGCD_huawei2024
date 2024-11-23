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
from unittest import TestCase, mock

from paho.mqtt.client import MQTTMessage

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.client.request.command_response import CommandRsp
from iot_device_sdk_python.iot_device import IotDevice
from iot_device_sdk_python.service.abstract_service import AbstractService
from iot_device_sdk_python.service.property import Property


class SmokeDetectorService(AbstractService):
    """
    烟感服务，支持属性：报警标志、烟雾浓度、温度、湿度
    支持的命令：响铃报警
    """

    def __init__(self, result_dict: dict):
        super().__init__()

        self.result_dict = result_dict

        # 按照设备模型定义属性，注意属性的prop_name需要和设备模型一致，writeable表示属性是否可写；field_name为变量的名字，val为属性的值
        self.smoke_alarm = Property(val=20, field_name="smoke_alarm", prop_name="alarm", writeable=True)
        self.concentration = Property(val=float(32.0), field_name="concentration", prop_name="smokeConcentration",
                                      writeable=False)
        self.humidity = Property(val=64, field_name="humidity", prop_name="humidity", writeable=False)
        self.temperature = Property(val=float(36.0), field_name="temperature", prop_name="temperature", writeable=False)
        # 定义命令名称与方法名称的映射关系
        self.command2method = {"ringAlarm": "alarm"}

        self.__set_writeable_and_readable(self.smoke_alarm, self.concentration, self.humidity, self.temperature)
        self.__set_command2method(self.command2method)

    def __set_writeable_and_readable(self, *args):
        for arg in args:
            self._readable_prop2field[arg.prop_name] = arg.field_name
            if arg.writeable:
                self._writeable_prop2field[arg.prop_name] = arg.field_name

    def __set_command2method(self, c2m):
        self._command2method = c2m

    # 定义命令，注意接口入参和返回值类型是固定的不能修改，否则会出现运行时错误
    def alarm(self, paras: dict):
        duration = paras.get("duration")
        command_rsp = CommandRsp()
        command_rsp.result_code = CommandRsp.success_code()
        self.result_dict["duration"] = duration
        return command_rsp

    # get和set接口的命名规则：get_ + 属性的变量名；设置正确，SDK会自动调用这些接口
    def get_humidity(self):
        # 模拟从传感器读取数据
        self.humidity.val = 32
        return self.humidity.val

    def set_humidity(self, humidity):
        # humidity是只读的，不需要实现
        pass

    def get_temperature(self):
        # 模拟从传感器读取数据
        self.temperature.val = 64
        return self.temperature.val

    def set_temperature(self, temperature):
        # 只读字段不需要实现set接口
        pass

    def get_concentration(self):
        # 模拟从传感器读取数据
        self.concentration.val = 36
        return self.concentration.val

    def set_concentration(self, concentration):
        # 只读字段不需要实现set接口
        pass

    def get_smoke_alarm(self):
        return self.smoke_alarm.val

    def set_smoke_alarm(self, smoke_alarm: int):
        self.smoke_alarm.val = smoke_alarm
        self.result_dict["alarm"] = smoke_alarm
        if smoke_alarm == 10:
            self._logger.info("alarm is clear by app")


class TestCustomServiceMethod(TestCase):
    result_dict = dict()

    @mock.patch("paho.mqtt.client.Client.is_connected")
    @mock.patch("paho.mqtt.client.Client.loop_forever")
    @mock.patch("paho.mqtt.client.Client.connect")
    def setUp(self, connect, loop_forever, is_connected):
        # mock
        connect.return_value = 0
        loop_forever.return_value = 0
        is_connected.return_value = True

        # setup
        server_uri = "access address"
        port = 8883
        self.device_id = "productId_nodeId"
        secret = "12345678"
        iot_ca_cert_path = "./resources/root.pem"

        connect_auth_info = ConnectAuthInfo()
        connect_auth_info.server_uri = server_uri
        connect_auth_info.port = port
        connect_auth_info.id = self.device_id
        connect_auth_info.secret = secret
        connect_auth_info.iot_cert_path = iot_ca_cert_path
        connect_auth_info.auth_type = ConnectAuthInfo.SECRET_AUTH
        connect_auth_info.bs_mode = ConnectAuthInfo.BS_MODE_DIRECT_CONNECT

        client_conf = ClientConf(connect_auth_info)

        self.device = IotDevice(client_conf)
        self.smoke_detector_service = SmokeDetectorService(self.result_dict)
        self.device.add_service("smokeDetector", self.smoke_detector_service)

        self.device.connect()
        return

    def test_custom_service_command(self):
        # test
        _get_connection = getattr(self.device.get_client(), "_get_connection")
        _connection = _get_connection()
        _get_paho_client = getattr(_connection, "_get_paho_client")
        _paho_client = _get_paho_client()
        _handle_on_message = getattr(_paho_client, "_handle_on_message")
        message = MQTTMessage()
        req_id = "123456"
        message.topic = b'$oc/devices/' \
                        + self.device_id.encode(encoding="utf-8") \
                        + b'/sys/commands/request_id=' \
                        + req_id.encode(encoding="utf-8")
        message.payload = b'{"object_device_id": "' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'","command_name": "ringAlarm","service_id": "smokeDetector","paras": {"duration": 123}}'
        _handle_on_message(message)

        # assert
        self.assertEqual(self.result_dict["duration"], 123)
        return

    def test_custom_service_property_set(self):
        # test
        _get_connection = getattr(self.device.get_client(), "_get_connection")
        _connection = _get_connection()
        _get_paho_client = getattr(_connection, "_get_paho_client")
        _paho_client = _get_paho_client()
        _handle_on_message = getattr(_paho_client, "_handle_on_message")
        message = MQTTMessage()
        req_id = "123456"
        message.topic = b'$oc/devices/' \
                        + self.device_id.encode(encoding="utf-8") \
                        + b'/sys/properties/set/request_id=' \
                        + req_id.encode(encoding="utf-8")
        message.payload = b'{"object_device_id": "' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'","services": [{"service_id": "smokeDetector", "properties": {"alarm": 36}}]}'
        _handle_on_message(message)

        # assert
        self.assertEqual(self.result_dict["alarm"], 36)
        return

