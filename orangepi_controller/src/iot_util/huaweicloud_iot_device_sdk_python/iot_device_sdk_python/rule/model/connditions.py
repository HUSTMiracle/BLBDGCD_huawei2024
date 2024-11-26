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

from typing import Optional
from typing import List
from iot_device_sdk_python.rule.model.device_info import DeviceInfo


class Condition:
    def __init__(self):
        self._type: str = ""
        self._start_time: str = ""
        self._repeat_interval: int = 0
        self._repeat_count: int = 0
        self._time: str = ""
        self._days_of_week: str = ""
        self._operator: str = ""
        self._device_info: Optional[DeviceInfo] = None
        self._value: str = ""
        self._in_values: List[str] = list()

    @property
    def type(self):
        """
        条件类型
        """
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def start_time(self):
        """
        开始时间
        """
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        self._start_time = value

    @property
    def repeat_interval(self):
        """
        执行周期
        """
        return self._repeat_interval

    @repeat_interval.setter
    def repeat_interval(self, value):
        self._repeat_interval = value

    @property
    def repeat_count(self):
        """
        执行次数
        """
        return self._repeat_count

    @repeat_count.setter
    def repeat_count(self, value):
        self._repeat_count = value

    @property
    def time(self):
        """
        执行时间
        """
        return self._time

    @time.setter
    def time(self, value):
        self._time = value

    @property
    def days_of_week(self):
        """
        日期
        """
        return self._days_of_week

    @days_of_week.setter
    def days_of_week(self, value):
        self._days_of_week = value

    @property
    def operator(self):
        """
        执行操作
        """
        return self._operator

    @operator.setter
    def operator(self, value):
        self._operator = value

    @property
    def device_info(self):
        """
        设备信息
        """
        return self._device_info

    @device_info.setter
    def device_info(self, value):
        self._device_info = value

    @property
    def value(self):
        """
        条件值
        """
        return self._value

    @value.setter
    def value(self, params):
        self._value = params

    @property
    def in_values(self):
        """
        条件值， operator为in时使用
        """
        return self._in_values

    @in_values.setter
    def in_values(self, value):
        self._in_values = value
