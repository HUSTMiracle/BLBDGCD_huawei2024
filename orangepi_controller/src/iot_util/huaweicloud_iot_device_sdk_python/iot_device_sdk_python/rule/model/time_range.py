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


class TimeRange:
    def __init__(self):
        self._start_time: str = ""
        self._end_time: str = ""
        self._days_of_week: str = ""

    @property
    def start_time(self):
        """
        设备的服务ID，在设备关联的产品模型中定义
        """
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        self._start_time = value

    @property
    def end_time(self):
        """
        设备的服务ID，在设备关联的产品模型中定义
        """
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        self._end_time = value

    @property
    def days_of_week(self):
        """
        设备的服务ID，在设备关联的产品模型中定义
        """
        return self._days_of_week

    @days_of_week.setter
    def days_of_week(self, value):
        self._days_of_week = value