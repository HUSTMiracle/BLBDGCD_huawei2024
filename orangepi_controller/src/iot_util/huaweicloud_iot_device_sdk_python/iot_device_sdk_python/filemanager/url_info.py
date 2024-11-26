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


class UrlInfo:
    def __init__(self):
        self._url: str = ''
        self._bucket_name: str = ''
        self._object_name: str = ''
        self._expire: int = 0
        self._file_attributes: dict = dict()

    @property
    def url(self):
        """
        文件上传/下载URL
        """
        return self._url

    @url.setter
    def url(self, value):
        self._url = value

    @property
    def bucket_name(self):
        """
        OBS桶的名称
        """
        return self._bucket_name

    @bucket_name.setter
    def bucket_name(self, value):
        self._bucket_name = value

    @property
    def object_name(self):
        """
        OBS待上传对象名称/OBS待下载对象名称
        """
        return self._object_name

    @object_name.setter
    def object_name(self, value):
        self._object_name = value

    @property
    def expire(self):
        """
        URL过期时间，单位：秒
        """
        return self._expire

    @expire.setter
    def expire(self, value):
        self._expire = value

    @property
    def file_attributes(self):
        """
        文件属性，JSON格式的字典
        """
        return self._file_attributes

    @file_attributes.setter
    def file_attributes(self, value):
        self._file_attributes = value

    def to_dict(self):
        return {"url": self._url, "bucket_name": self._bucket_name, "object_name": self._object_name,
                "expire": self._expire, "file_attributes": self._file_attributes}

    def convert_from_dict(self, json_dict: dict):
        json_name = ["url", "bucket_name", "object_name", "expire"]
        for key in json_dict.keys():
            if key not in json_name:
                continue
            if key == "url":
                self.url = json_dict.get(key)
            elif key == "bucket_name":
                self.bucket_name = json_dict.get(key)
            elif key == "object_name":
                self.object_name = json_dict.get(key)
            elif key == "expire":
                self.expire = json_dict.get(key)
            elif key == "file_attributes":
                self.file_attributes = json_dict.get(key)
            else:
                pass

