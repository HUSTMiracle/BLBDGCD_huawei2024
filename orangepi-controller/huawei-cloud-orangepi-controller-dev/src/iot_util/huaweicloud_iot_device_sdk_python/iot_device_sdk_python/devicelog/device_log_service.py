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
from typing import Optional
import logging
import time

from iot_device_sdk_python.service.abstract_service import AbstractService
from iot_device_sdk_python.client.request.device_event import DeviceEvent
from iot_device_sdk_python.utils.iot_util import get_event_time


class DeviceLogService(AbstractService):
    _logger = logging.getLogger(__name__)
    _LOG_CONFIG = "log_config"

    def __init__(self):
        super().__init__()
        self._log_switch = True  # 默认为True
        self._end_time = None
        self._connect_lost_dict: Optional[dict] = None
        self._connect_failed_dict: Optional[dict] = None

    def on_event(self, device_event: DeviceEvent):
        """
        设备日志服务的事件处理方法

        Args:
            device_event: 设备事件
        """
        if device_event.event_type == self._LOG_CONFIG:
            # 平台下发日志收集通知
            paras: dict = device_event.paras
            if "switch" in paras.keys():
                str_switch = paras.get("switch")
            else:
                self._logger.warning("event.paras doesn't contain key: switch. paras: %s", str(paras))
                return
            if "end_time" in paras.keys():
                end_time = paras.get("end_time")
                self.end_time = end_time
            else:
                self._logger.debug("event.paras doesn't contain key: end_time, paras: %s", str(paras))
            if str_switch == "on":
                self.log_switch = True
            elif str_switch == "off":
                self.log_switch = False

    def report_device_log(self, timestamp: str, log_type: str, content: str):
        """
        设备上报日志内容

        Args:
            timestamp:   日志产生的时间戳，精确到秒
            log_type:    日志类型，总共有如下几种：

                                DEVICE_STATUS:      设备状态

                                DEVICE_PROPERTY:    设备属性

                                DEVICE_MESSAGE:     设备消息

                                DEVICE_COMMAND:     设备命令
            content:     日志内容
        """
        device_event = DeviceEvent()
        device_event.service_id = self.service_id
        device_event.event_type = "log_report"
        device_event.event_time = get_event_time()
        paras: dict = {"timestamp": timestamp,
                       "type": log_type,
                       "content": content}
        device_event.paras = paras
        self.get_iot_device().get_client().report_event(device_event)

    def can_report_log(self):
        """
        根据平台上设置的开关和结束时间来判断能否上报日志

        Returns:
            bool: True为能上报日志；False为不具备上报的条件;
        """
        end_time: str = self.end_time
        if end_time is not None:
            end_time = end_time.replace("T", "")
            end_time = end_time.replace("Z", "")

        current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

        if self.log_switch and (end_time is None or current_time < end_time):
            return True
        return False

    @property
    def log_switch(self):
        return self._log_switch

    @log_switch.setter
    def log_switch(self, value):
        self._log_switch = value

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        self._end_time = value

    @property
    def connect_lost_dict(self):
        return self._connect_lost_dict

    @connect_lost_dict.setter
    def connect_lost_dict(self, value):
        self._connect_lost_dict = value

    @property
    def connect_failed_dict(self):
        return self._connect_failed_dict

    @connect_failed_dict.setter
    def connect_failed_dict(self, value):
        self._connect_failed_dict = value
