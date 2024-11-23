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

import threading
import time

import schedule
from schedule import Job

from iot_device_sdk_python.rule.model.actions import Action
from iot_device_sdk_python.rule.model.connditions import Condition
from iot_device_sdk_python.rule.model.device_info import DeviceInfo
from iot_device_sdk_python.rule.model.rule_info import RuleInfo
from iot_device_sdk_python.rule.model.time_range import TimeRange
from iot_device_sdk_python.client.device_client import DeviceClient
from iot_device_sdk_python.transport.raw_message import RawMessage
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from iot_device_sdk_python.rule.util.time_util import get_rule_week, check_time_range
from typing import List
from datetime import timedelta, timezone
import logging
import os
import datetime
import threading


class TimerRuleInstance:
    _logger = logging.getLogger(__name__)

    def __init__(self, device_client: DeviceClient):
        super().__init__()
        self.__device_client = device_client
        self.__rule_status = False
        self.__job_list: List[Job] = list()
        self.__background_schedule: BackgroundScheduler = BackgroundScheduler()

    def submit_rule(self, rule_info: RuleInfo):
        condition_list: List[Condition] = rule_info.conditions
        for condition in condition_list:
            timer_type = condition.type
            if "DAILY_TIMER" == timer_type:
                execute_time = condition.time
                days_of_week = condition.days_of_week
                if not execute_time or not days_of_week:
                    self._logger.warning("time or days of week is empty. time: %s, days of week: %s", execute_time,
                                         days_of_week)
                    continue
                time_list: List[int] = list(map(int, execute_time.split(":")))
                if len(time_list) != 2:
                    self._logger.warning("time format is invalid. time: %s", execute_time)
                    continue
                week_list: List[int] = list(map(get_rule_week, days_of_week.split(",")))
                for i in range(len(week_list)):
                    schedule_week: int = week_list[i]
                    self.__background_schedule.add_job(self.__execute_rule,
                                                       args=[rule_info.actions, rule_info.time_range], trigger="cron",
                                                       day_of_week=str(schedule_week), hour=time_list[0],
                                                       minute=time_list[1], timezone=timezone.utc)
            elif "SIMPLE_TIMER" == timer_type:
                repeat_interval = condition.repeat_interval
                repeat_count = condition.repeat_count
                begin = datetime.datetime.strptime(condition.start_time, "%Y-%m-%d %H:%M:%S")
                end = begin + timedelta(seconds=(repeat_count - 1) * repeat_interval)
                self.__background_schedule.add_job(self.__execute_rule, args=[rule_info.actions, rule_info.time_range],
                                                   trigger="interval",
                                                   start_date=begin, end_date=end, seconds=repeat_interval,
                                                   timezone=timezone.utc)

    def start(self):
        self.__background_schedule.start()

    def __execute_rule(self, action: List[Action], timerange: TimeRange):
        if check_time_range(timerange):
            self.__device_client.on_rule_action_handler(action)

    def shutdown_timer(self):
        self._logger.info("shut down schedule.")
        if self.__background_schedule.state == 1:
            self.__background_schedule.shutdown()


def execute_task():
    print("execute task")
