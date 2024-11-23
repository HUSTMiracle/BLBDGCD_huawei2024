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
from iot_device_sdk_python.rule.model.actions import Action
from iot_device_sdk_python.rule.model.connditions import Condition
from iot_device_sdk_python.rule.model.time_range import TimeRange


class RuleInfo:

    def __init__(self):
        self._rule_id: str = ""
        self._rule_name: str = ""
        self._logic: str = ""
        self._time_range: Optional[TimeRange] = None
        self._status: str = ""
        self._conditions: List[Condition] = list()
        self._actions: List[Action] = list()
        self._rule_version_in_shadow: int = 0

    @property
    def rule_id(self):
        """
        设备的服务ID，在设备关联的产品模型中定义
        """
        return self._rule_id

    @rule_id.setter
    def rule_id(self, value):
        self._rule_id = value

    @property
    def rule_name(self):
        """
        设备的服务ID，在设备关联的产品模型中定义
        """
        return self._rule_name

    @rule_name.setter
    def rule_name(self, value):
        self._rule_name = value

    @property
    def logic(self):
        """
        设备的服务ID，在设备关联的产品模型中定义
        """
        return self._logic

    @logic.setter
    def logic(self, value):
        self._logic = value

    @property
    def time_range(self):
        """
        设备的服务ID，在设备关联的产品模型中定义
        """
        return self._time_range

    @time_range.setter
    def time_range(self, value):
        self._time_range = value

    @property
    def status(self):
        """
        设备的服务ID，在设备关联的产品模型中定义
        """
        return self._status

    @status.setter
    def status(self, value):
        self._status = value

    @property
    def conditions(self):
        """
        设备的服务ID，在设备关联的产品模型中定义
        """
        return self._conditions

    @conditions.setter
    def conditions(self, value):
        self._conditions = value

    @property
    def actions(self):
        """
        设备的服务ID，在设备关联的产品模型中定义
        """
        return self._actions

    @actions.setter
    def actions(self, value):
        self._actions = value

    @property
    def rule_version_in_shadow(self):
        """
        设备的服务ID，在设备关联的产品模型中定义
        """
        return self._rule_version_in_shadow

    @rule_version_in_shadow.setter
    def rule_version_in_shadow(self, value):
        self._rule_version_in_shadow = value
