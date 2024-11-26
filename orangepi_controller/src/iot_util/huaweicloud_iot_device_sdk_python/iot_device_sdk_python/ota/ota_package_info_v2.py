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


class OTAPackageInfoV2:
    def __init__(self):
        self._url: str = ""
        self._version: str = ""
        self._file_size: Optional[int] = None
        self._file_name: str = None
        self._expires: Optional[int] = None
        self._sign: str = None
        self._custom_info: str = None
        self._task_id: str = None
        self._sub_device_count: Optional[int] = None
        self._task_ext_info: Union[str, dict] = None

    @property
    def url(self):
        """
        软固件包下载地址
        """
        return self._url

    @url.setter
    def url(self, value):
        self._url = value

    @property
    def version(self):
        """
        软固件包版本号
        """
        return self._version

    @version.setter
    def version(self, value):
        self._version = value

    @property
    def file_size(self):
        """
        软固件包文件大小
        """
        return self._file_size

    @file_size.setter
    def file_size(self, value):
        self._file_size = value

    @property
    def file_name(self):
        """
        软固件包名称
        """
        return self._file_name

    @file_name.setter
    def file_name(self, value):
        self._file_name = value

    @property
    def expires(self):
        """
        access_token的超期时间
        """
        return self._expires

    @expires.setter
    def expires(self, value):
        self._expires = value

    @property
    def sign(self):
        """
        软固件包SHA-256值
        """
        return self._sign

    @sign.setter
    def sign(self, value):
        self._sign = value

    @property
    def custom_info(self):
        """
        下发的包的自定义信息
        """
        return self._custom_info

    @custom_info.setter
    def custom_info(self, value):
        self._custom_info = value

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
    def sub_device_count(self):
        """
        网关模式下，升级中子设备数量
        """
        return self._sub_device_count

    @sub_device_count.setter
    def sub_device_count(self, value):
        self._sub_device_count = value

    @property
    def task_ext_info(self):
        """
        批量升级任务额外扩展信息
        """
        return self._task_ext_info

    @task_ext_info.setter
    def task_ext_info(self, value):
        self._task_ext_info = value

    def to_dict(self):
        dict_info = {"url": self._url, "version": self._version, "file_size": self._file_size,
                     "file_name": self._file_name, "expires": self._expires}
        if self._custom_info is not None:
            dict_info["custom_info"] = self._custom_info
        if self._sign is not None:
            dict_info["sign"] = self._sign
        if self._task_id is not None:
            dict_info["task_id"] = self._task_id
        if self._sub_device_count is not None:
            dict_info["sub_device_count"] = self._sub_device_count
        if self._task_ext_info is not None:
            dict_info["task_ext_info"] = self._task_ext_info
        return dict_info

    def convert_from_dict(self, json_dict: dict):
        json_name = ["url", "version", "expires"]
        for key in json_dict.keys():
            if key not in json_name:
                continue
            if key == "url":
                self.url = json_dict.get(key)
            elif key == "version":
                self.version = json_dict.get(key)
            elif key == "expires":
                self.expires = json_dict.get(key)
            elif key == "file_size":
                self.file_size = json_dict.get(key)
            elif key == "file_name":
                self.file_name = json_dict.get(key)
            elif key == "sign":
                self.sign = json_dict.get(key)
            elif key == "custom_info":
                self.custom_info = json_dict.get(key)
            elif key == "task_id":
                self.task_id = json_dict.get(key)
            elif key == "sub_device_count":
                self.sub_device_count = json_dict.get(key)
            elif key == "task_ext_info":
                self.task_ext_info = json_dict.get(key)
            else:
                pass

