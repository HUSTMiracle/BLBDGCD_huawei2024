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

from __future__ import absolute_import, annotations
from typing import List, TYPE_CHECKING
import logging
import threading
import traceback
import time

import schedule

from iot_device_sdk_python.service.i_service import IService
from iot_device_sdk_python.client.iot_result import IotResult
from iot_device_sdk_python.client.iot_result import SUCCESS
from iot_device_sdk_python.client.request.command import Command
from iot_device_sdk_python.client.request.command_response import CommandRsp

if TYPE_CHECKING:
    from iot_device_sdk_python.service.abstract_device import AbstractDevice


class AbstractService(IService):
    """
    抽象服务类，提供了属性自动读写和命令调用能力，用户可以继承此类，根据物模型定义自己的服务
    """
    _logger = logging.getLogger(__name__)

    def __init__(self):
        self._iot_device = None
        self._service_id = ""
        self._writeable_prop2field = dict()
        self._readable_prop2field = dict()
        self._command2method = dict()
        self._auto_report = False
        self._thread = None

    def on_read(self, properties: List[str]):
        """
        读属性回调

        Args:
            properties:  指定读取的属性名列表，若列表为[]则读取全部可读属性
        Returns:
            dict:    属性值
        """
        ret = dict()
        if not properties:
            self._logger.debug("read all properties")
        else:
            self._logger.debug("read properties: %s", str(properties))
        # 读取指定的字段
        if len(properties) > 0:
            for prop_name in properties:
                # prop_name是产品模型中定义的属性名称

                if prop_name not in self._readable_prop2field.keys():
                    self._logger.debug("property is not readable: %s", prop_name)
                    continue
                # field_name是prop_name在代码中所对应的变量名称
                field_name = self._readable_prop2field.get(prop_name)
                if not hasattr(self, field_name):
                    self._logger.debug("field not found: %s", field_name)
                    continue
                getter = "get_" + field_name
                if not hasattr(self, getter):
                    self._logger.debug("getter method not found: %s", getter)
                    continue
                get_method = getattr(self, getter)
                try:
                    prop_val = get_method()
                except Exception as e:
                    self._logger.warning("getter method: %s failed, exception: %s", get_method, str(e))
                    continue
                ret[prop_name] = prop_val
            return ret

        # 读取全部字段
        for prop_name, field_name in self._readable_prop2field.items():
            if not hasattr(self, field_name):
                self._logger.debug("field not found: %s", field_name)
                continue
            getter = "get_" + field_name
            if not hasattr(self, getter):
                self._logger.debug("getter method not found: %s", getter)
                continue
            get_method = getattr(self, getter)
            try:
                prop_val = get_method()
            except Exception as e:
                self._logger.warning("getter method: %s failed, traceback: %s", get_method, str(e))
                continue
            ret[prop_name] = prop_val
        return ret

    def on_write(self, properties: dict) -> IotResult:
        """
        写属性回调。收到平台下发的写属性操作时此接口被自动调用。
        如果用户希望在写属性时增加额外处理，可以重写此接口。

        Args:
            properties:  平台期望属性的值
        Returns:
            IotResult:    操作结果
        """
        changed_props = list()
        for k, v in properties.items():
            # prop_name, value
            if k not in self._writeable_prop2field.keys():
                self._logger.warning("property not found or not writeable: %s", k)
                return IotResult(-1, "property not found or not writeable: %s" % k)
            if not hasattr(self, self._writeable_prop2field.get(k)):
                self._logger.warning("field not found: %s", k)
                return IotResult(-1, "field not found: %s" % k)
            field_name = self._writeable_prop2field.get(k)
            setter = "set_" + field_name
            if not hasattr(self, setter):
                self._logger.warning("setter method not found: %s", setter)
                return IotResult(-1, "setter method not found: %s" % setter)
            set_method = getattr(self, setter)
            try:
                set_method(v)
                changed_props.append(k)
            except Exception as e:
                self._logger.warning("setter method: %s failed, property: %s, field: %s, traceback: %s", set_method, k,
                                     field_name,
                                     traceback.format_exc())
                return IotResult(-1, str(e))

        if len(changed_props) > 0:
            self.fire_properties_changed(changed_props)
        return SUCCESS

    def on_event(self, device_event):
        """
        事件处理回调。收到平台下发的事件时此接口被自动调用。默认为空实现，由子类服务实现。

        Args:
            device_event: 设备事件
        """
        self._logger.debug("AbstractService on_event method has no operation")

    def fire_properties_changed(self, properties: List[str]):
        """
        通知服务属性变化

        Args:
            properties:  变化的属性
        """
        self.get_iot_device().fire_properties_changed(self._service_id, properties)

    def on_command(self, command: Command) -> CommandRsp:
        """
        执行设备命令。收到平台下发的命令时此接口被自动调用

        Args:
            command: 命令请求
        Returns:
            CommandRsp: 命令响应
        """
        command_name = command.command_name
        if command_name not in self._command2method.keys():
            self._logger.debug("command not found: %s", command_name)
            command_rsp = CommandRsp()
            command_rsp.result_code = CommandRsp.fail_code()
            command_rsp.paras = {"error": "command not found: %s" % command_name}
            return command_rsp
        method_name = self._command2method.get(command_name)
        if not hasattr(self, method_name):
            self._logger.debug("method not found: %s", method_name)
            command_rsp = CommandRsp()
            command_rsp.result_code = CommandRsp.fail_code()
            command_rsp.paras = {"error": "method not found: %s" % method_name}
            return command_rsp
        method = getattr(self, method_name)
        try:
            command_rsp: CommandRsp = method(command.paras)
        except Exception as e:
            self._logger.warning("command execute failed, command: %s, method: %s, traceback: %s", command_name,
                                 method_name,
                                 traceback.format_exc())
            command_rsp = CommandRsp()
            command_rsp.result_code = CommandRsp.fail_code()
            command_rsp.paras = {"error": "method execute failed, Exception: %s" % str(e)}
            return command_rsp

        return command_rsp

    def get_iot_device(self) -> AbstractDevice:
        """
        获取设备实例
        """
        if self._iot_device is None:
            self._logger.error("IotDevice in %s is None, "
                               "please call set_iot_device() to set an IotDevice, "
                               "return None", self._service_id)
        return self._iot_device

    def set_iot_device(self, iot_device: AbstractDevice):
        """
        设置设备实例
        """
        self._iot_device = iot_device

    @property
    def service_id(self):
        return self._service_id

    @service_id.setter
    def service_id(self, value):
        self._service_id = value

    def enable_auto_report(self, report_interval: int):
        """
        开启自动周期上报属性

        Args:
            report_interval (int): 上报周期，单位为秒
        """
        if self._auto_report:
            self._logger.warning("timer is already enable")
            return
        else:
            self._auto_report = True
            self._thread = threading.Thread(target=self._auto_report_thread_func, args=(report_interval,))
            self._thread.start()

    def disable_auto_report(self):
        """
        关闭自动周期上报，用户可以通过fire_properties_changed触发上报
        """
        if self._auto_report:
            self._auto_report = False

    def _auto_report_thread_func(self, report_interval: int):
        """
        周期上报属性方法

        Args:
            report_interval: 上报周期，单位s
        """
        schedule.every(report_interval).seconds.do(self.fire_properties_changed, [])
        while self._auto_report:
            schedule.run_pending()
            time.sleep(1)
