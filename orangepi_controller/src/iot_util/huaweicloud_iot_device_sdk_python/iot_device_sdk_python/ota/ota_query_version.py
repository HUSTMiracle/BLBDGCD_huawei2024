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
from typing import Optional, Union


class OTAQueryVersion:
    def __init__(self):
        self._task_id: str = None
        self._task_ext_info: Union[str, dict] = None
        self._sub_device_count: Optional[int] = None

    @property
    def task_id(self):
        """
        网关模式下，创建升级任务的任务ID
        """
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        self._task_id = value

    @property
    def task_ext_info(self):
        """
        批量升级任务额外扩展信息
        """
        return self._task_ext_info

    @task_ext_info.setter
    def task_ext_info(self, value):
        self._task_ext_info = value

    @property
    def sub_device_count(self):
        """
        网关模式下，升级中子设备数量
        """
        return self._sub_device_count

    @sub_device_count.setter
    def sub_device_count(self, value):
        self._sub_device_count = value

    def to_dict(self):
        dict_info = {}
        if self._task_id is not None:
            dict_info["task_id"] = self._task_id
        if self._task_ext_info is not None:
            dict_info["task_ext_info"] = self._task_ext_info
        if self._sub_device_count is not None:
            dict_info["sub_device_count"] = self._sub_device_count
        return dict_info

    def convert_from_dict(self, json_dict: dict):
        json_name = ["task_id", "sub_device_count", "task_ext_info"]
        for key in json_dict.keys():
            if key not in json_name:
                continue
            if key == "task_id":
                self.task_id = json_dict.get(key)
            elif key == "task_ext_info":
                self.task_ext_info = json_dict.get(key)
            elif key == "sub_device_count":
                self.sub_device_count = json_dict.get(key)
            else:
                pass
