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
此例用来：
演示面向物模型编程的方法。
    用户只需要根据物模型定义自己的设备服务类，就可以直接对设备服务进行读写操作，
    SDK会自动的完成设备属性的同步和命令的调用。本例中实现的设备服务为烟感服务。
"""

import time
import logging

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.service.abstract_service import AbstractService
from iot_device_sdk_python.service.property import Property
from iot_device_sdk_python.client.request.command_response import CommandRsp
from iot_device_sdk_python.iot_device import IotDevice


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(threadName)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")


class SmokeDetector:
    _logger = logging.getLogger(__name__)

    def __init__(self, server_uri, port, device_id, secret, iot_cert_file):
        self.server_uri = server_uri
        self.port = port
        self.device_id = device_id
        self.secret = secret
        self.iot_cert_file = iot_cert_file

    def start(self):
        """ 创建设备 """
        connect_auth_info = ConnectAuthInfo()
        connect_auth_info.server_uri = self.server_uri
        connect_auth_info.port = self.port
        connect_auth_info.id = self.device_id
        connect_auth_info.secret = self.secret
        connect_auth_info.iot_cert_path = self.iot_cert_file
        connect_auth_info.bs_mode = ConnectAuthInfo.BS_MODE_DIRECT_CONNECT

        client_conf = ClientConf(connect_auth_info)

        device = IotDevice(client_conf)

        """ 添加烟感服务 """
        smoke_detector_service = SmokeDetectorService()
        device.add_service("smokeDetector", smoke_detector_service)

        """ 设备连接平台 """
        if device.connect() != 0:
            return

        """ 启动自动周期上报 """
        smoke_detector_service.enable_auto_report(5)

        """ 20s后结束周期上报 """
        time.sleep(20)
        smoke_detector_service.disable_auto_report()


class SmokeDetectorService(AbstractService):
    """
    烟感服务，支持属性：报警标志、烟雾浓度、温度、湿度
    支持的命令：响铃报警
    """
    _logger = logging.getLogger(__name__)

    def __init__(self):
        super().__init__()

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
        self._logger.info("ringAlarm duration = " + str(duration))
        command_rsp = CommandRsp()
        command_rsp.result_code = CommandRsp.success_code()
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
        if smoke_alarm == 10:
            self._logger.info("set alarm:" + str(smoke_alarm))
            self._logger.info("alarm is clear by app")


def run():
    # 替换为真正的接入地址、device id和密钥，参考readme_CN.md中“上传产品模型并注册设备”
    server_uri = "access address"
    port = 8883
    device_id = "your device id"
    secret = "your device secret"
    iot_ca_cert_path = "./resources/root.pem"

    smoke_detector = SmokeDetector(server_uri=server_uri,
                                   port=port,
                                   device_id=device_id,
                                   secret=secret,
                                   iot_cert_file=iot_ca_cert_path)
    smoke_detector.start()


if __name__ == "__main__":
    run()
