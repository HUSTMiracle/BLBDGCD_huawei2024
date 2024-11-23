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
from datetime import datetime
from typing import Optional
from typing import List
import logging
import os
import uuid

from iot_device_sdk_python.client.request.device_event import DeviceEvent
from iot_device_sdk_python.client.request.device_events import DeviceEvents
from iot_device_sdk_python.service.abstract_service import AbstractService
from iot_device_sdk_python.filemanager.url_info import UrlInfo
from iot_device_sdk_python.filemanager.file_manager_listener import FileManagerListener
from iot_device_sdk_python.transport.action_listener import ActionListener
from iot_device_sdk_python.utils.iot_util import get_event_time
from iot_device_sdk_python.utils.iot_util import sha256_hash_from_file
from iot_device_sdk_python.rule.model.actions import Action
from iot_device_sdk_python.rule.model.connditions import Condition
from iot_device_sdk_python.rule.model.device_info import DeviceInfo
from iot_device_sdk_python.rule.model.rule_info import RuleInfo
from iot_device_sdk_python.rule.model.time_range import TimeRange
from iot_device_sdk_python.rule.timer_rule.timer_rule import TimerRuleInstance
from iot_device_sdk_python.rule.model.condition_execute import ConditionExecute
from iot_device_sdk_python.rule.model.action_handler import ActionHandler
from iot_device_sdk_python.client.request.command import Command
from iot_device_sdk_python.client.iot_result import IotResult
from iot_device_sdk_python.client.iot_result import SUCCESS
from iot_device_sdk_python.client.request.service_property import ServiceProperty
from iot_device_sdk_python.utils.iot_util import get_event_time
from iot_device_sdk_python.client.listener.command_listener import CommandListener
from iot_device_sdk_python.client.device_client import DeviceClient
from iot_device_sdk_python.transport.raw_message import RawMessage
from iot_device_sdk_python.rule.util.time_util import check_time_range


class RuleManageService(AbstractService):
    """
    规则管理器
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, device_client: DeviceClient):
        super().__init__()
        self._rule_id_list: Optional[list] = list()
        self._rule_info: dict = dict()
        self._timer_rule_dict: dict = dict()
        self._device_client: DeviceClient = device_client
        self._condition_execute: ConditionExecute = ConditionExecute()

    def on_event(self, device_event: DeviceEvent):
        """
        端侧规则服务处理
        Args:
            device_event:    事件
        """
        if not self._device_client.enable_rule_manage():
            self._logger.warning("rule manage enable is false. no need to resolve rule.")
            return
        service_id = device_event.service_id
        event_type = device_event.event_type
        if service_id != "$device_rule" and event_type != "device_rule_config_response":
            self._logger.warning("service id %s is not match", service_id)
            return
        params: dict = device_event.paras
        rules_infos: List[dict] = params.get("rulesInfos")
        if len(rules_infos) <= 0:
            self._logger.warning("rules info length is %d", len(rules_infos))
            return
        for rule in rules_infos:
            rule_info = RuleInfo()
            stats = rule.get("status")
            rule_info.rule_id = rule.get("ruleId")
            if stats == "inactive":
                self._logger.info("rule is inactive.")
                if rule_info.rule_id in self._rule_id_list:
                    self._rule_id_list.remove(rule_info.rule_id)
                if rule_info.rule_id in self._rule_info.keys():
                    self._rule_info.pop(rule_info.rule_id)
                continue
            rule_info.rule_version_in_shadow = rule.get("ruleVersionInShadow")
            old_rule: RuleInfo = self._rule_info.get(rule_info.rule_id)
            if old_rule is not None and old_rule.rule_version_in_shadow >= rule_info.rule_version_in_shadow:
                self._logger.info("rule version is not change. no need to refresh.")
                continue
            rule_info.rule_name = rule.get("ruleName")
            rule_info.logic = rule.get("logic")
            time_range = TimeRange()
            time_range_dict: dict = rule.get("timeRange")
            if time_range_dict:
                time_range.start_time = time_range_dict.get("startTime")
                time_range.end_time = time_range_dict.get("endTime")
                time_range.days_of_week = time_range_dict.get("daysOfWeek")
                rule_info.time_range = time_range
            rule_info.status = rule.get("status")
            conditions: List[Condition] = list()
            condition_list: List[dict] = rule.get("conditions")
            for value in condition_list:
                condition = Condition()
                condition.type = value.get("type")
                condition.operator = value.get("operator")
                condition.start_time = value.get("startTime")
                condition.repeat_interval = value.get("repeatInterval")
                condition.repeat_count = value.get("repeatCount")
                condition.time = value.get("time")
                condition.days_of_week = value.get("daysOfWeek")
                device_info = DeviceInfo()
                info_msg: dict = value.get("deviceInfo")
                if info_msg:
                    device_info.device_id = info_msg.get("deviceId")
                    device_info.path = info_msg.get("path")
                    condition.device_info = device_info
                condition.value = value.get("value")
                condition.in_values = value.get("inValues")
                conditions.append(condition)

            rule_info.conditions = conditions
            actions: List[Action] = list()
            action_list: List[dict] = rule.get("actions")
            for value in action_list:
                action = Action()
                action.type = value.get("type")
                action.status = value.get("status")
                action.device_id = value.get("deviceId")
                command_msg: dict = value.get("command")
                command = Command()
                command.command_name = command_msg.get("commandName")
                command.paras = command_msg.get("commandBody")
                command.service_id = command_msg.get("serviceId")
                action.command = command
                actions.append(action)
            rule_info.actions = actions
            self._rule_info[rule_info.rule_id] = rule_info
            self._submit_timer_rule(rule_info)
        self._logger.info("rule info list is %s", self._rule_info)

    def on_write(self, properties: dict) -> IotResult:
        """
        Args:
            properties:  规则数据
        Returns:
            IotResult:    操作结果
        """
        if not self._device_client.enable_rule_manage():
            self._logger.warning("rule manage enable is false. no need to resolve rule.")
            return SUCCESS
        self._rule_id_list = list(set(self._rule_id_list))
        rule_id_del = list()
        for key in properties.keys():
            version_dict: dict = properties.get(key)
            version = version_dict.get("version")
            if version != -1:
                if key not in self._rule_id_list:
                    self._rule_id_list.append(key)
            else:
                if key in self._rule_id_list:
                    self._rule_id_list.remove(key)
                if key in self._rule_info.keys():
                    self._rule_info.pop(key)
                    timer_rule: TimerRuleInstance = self._timer_rule_dict.pop(key)
                    timer_rule.shutdown_timer()
                rule_id_del.append(key)
        device_event = DeviceEvent()
        if len(self._rule_id_list) <= 0:
            return SUCCESS

        device_event.service_id = "$device_rule"
        device_event.event_type = "device_rule_config_request"
        device_event.event_time = get_event_time()
        params: dict = {"ruleIds": self._rule_id_list, "delIds": rule_id_del}
        device_event.paras = params
        self.get_iot_device().get_client().report_event(device_event)
        return SUCCESS

    def handle_rule(self, services: List[ServiceProperty]):
        for key in self._rule_info.keys():
            try:
                rule: RuleInfo = self._rule_info.get(key)
                # 判断是否在时间范围内
                if not check_time_range(rule.time_range):
                    self._logger.warning("rule was not match the time.")
                    return
                conditions: List[Condition] = rule.conditions
                logic = rule.logic
                if logic == "or":
                    for condition in conditions:
                        flag = self._condition_execute.is_condition_satisfied(condition, services)
                        if flag:
                            self._device_client.on_rule_action_handler(rule.actions)
                            break
                elif logic == "and":
                    is_satisfied: bool = True
                    for condition in conditions:
                        flag = self._condition_execute.is_condition_satisfied(condition, services)
                        if not flag:
                            is_satisfied = False
                    if is_satisfied:
                        self._device_client.on_rule_action_handler(rule.actions)
                else:
                    self._logger.warning("rule logic is not match. logic: %s", logic)

            except Exception as e:
                self._logger.error("handle rule failed. e: %s", e)
                pass

    def handle_action(self, action: Action):
        command = action.command
        if command is None:
            self._logger.warning("rule command is None.")
            return
        self._device_client.on_rule_command(uuid.uuid4().hex, command)

    def _submit_timer_rule(self, rule_info: RuleInfo):
        condition_list: List[Condition] = rule_info.conditions
        is_timer_rule = False
        for condition in condition_list:
            if "DAILY_TIMER" == condition.type or "SIMPLE_TIMER" == condition.type:
                is_timer_rule = True
                break
        if not is_timer_rule:
            return
        if is_timer_rule and len(condition_list) > 1 and rule_info.logic == "and":
            self._logger.warning("multy timer rule only support or logic. ruleId: %s", key)
            return
        key = rule_info.rule_id
        if key in self._timer_rule_dict:
            timer_rule: TimerRuleInstance = self._timer_rule_dict.pop(key)
            timer_rule.shutdown_timer()

        timer_rule_instance = TimerRuleInstance(self._device_client)
        timer_rule_instance.submit_rule(rule_info)
        timer_rule_instance.start()
        self._timer_rule_dict[key] = timer_rule_instance
