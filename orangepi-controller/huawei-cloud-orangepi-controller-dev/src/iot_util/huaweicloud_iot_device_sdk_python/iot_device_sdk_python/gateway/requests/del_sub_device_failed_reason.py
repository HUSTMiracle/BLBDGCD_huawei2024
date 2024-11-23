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


class DelSubDeviceFailedReason:
    def __init__(self):
        self._device_id: str = ""
        self._error_code: str = ""
        self._error_msg: str = ""

    @property
    def device_id(self):
        return self._device_id

    @device_id.setter
    def device_id(self, value):
        self._device_id = value

    @property
    def error_code(self):
        return self._error_code

    @error_code.setter
    def error_code(self, value):
        self._error_code = value

    @property
    def error_msg(self):
        return self._error_msg

    @error_msg.setter
    def error_msg(self, value):
        self._error_msg = value

    def to_dict(self):
        return {"device_id": self._device_id,
                "error_code": self._error_code,
                "error_msg": self._error_msg}

    def convert_from_dict(self, json_dict: dict):
        json_name = ["device_id", "error_code", "error_msg"]
        for key in json_dict.keys():
            if key not in json_name:
                continue
            if key == "device_id":
                self.device_id = json_dict.get(key)
            elif key == "error_code":
                self.error_code = json_dict.get(key)
            elif key == "error_msg":
                self.error_msg = json_dict.get(key)
            else:
                pass












