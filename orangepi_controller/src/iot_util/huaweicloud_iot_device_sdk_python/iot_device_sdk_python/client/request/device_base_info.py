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

class DeviceBaseInfo:
    def __init__(self):
        self._fw_version: str = ""
        self._sw_version: str = ""

    @property
    def fw_version(self):
        """
        固件版本
        """
        return self._fw_version

    @fw_version.setter
    def fw_version(self, value):
        self._fw_version = value

    @property
    def sw_version(self):
        """
        软件版本
        """
        return self._sw_version

    @sw_version.setter
    def sw_version(self, value):
        self._sw_version = value
