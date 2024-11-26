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

from typing import List
from datetime import datetime
from iot_device_sdk_python.rule.model.time_range import TimeRange


def check_time_range(time_range: TimeRange) -> bool:
    if time_range is None:
        return True

    begin_time = time_range.start_time
    end_time = time_range.end_time
    week_str = time_range.days_of_week
    if begin_time is None or end_time is None or week_str is None:
        return False
    now = datetime.utcnow()
    week_list: List[int] = list(map(int, week_str.split(",")))
    #
    now_week = (now.weekday() + 2) % 7
    # 开始结束时间分割成[Hour, Minute]
    begin_time_list: List[int] = list(map(int, begin_time.split(":")))
    end_time_list: List[int] = list(map(int, end_time.split(":")))
    begin_hour = begin_time_list[0]
    begin_minute = begin_time_list[1]
    end_hour = end_time_list[0]
    end_minute = end_time_list[1]
    now_hour = now.hour
    now_minute = now.minute

    # 8:00 -9:00形式
    if (begin_hour * 60 + begin_minute) < (end_hour * 60 + end_minute):
        return ((begin_hour * 60 + begin_minute) <= (now_hour * 60 + now_minute) <= (
                end_hour * 60 + end_minute)) and now_week in week_list

    # 23:00 -01:00形式， 处于23:00-00:00之间的形式
    if (begin_hour * 60 + begin_minute) <= (now_hour * 60 + now_minute) <= (24 * 60 + 00) and now_week in week_list:
        return True
    elif (now_hour * 60 + now_minute) <= (end_hour * 60 + end_minute):
        now_week = now_week - 1
        if 0 == now_week:
            now_week = 7
        return now_week in week_list
    return False


def get_rule_week(week: str) -> int:
    cur_week = int(week) - 2
    if cur_week == -1:
        cur_week = 6
    return cur_week
