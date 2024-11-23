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

'''设备消息
'''


class DeviceMessage:
    def __init__(self, message=None):
        self.__object_device_id = None
        self.__id = None
        self.__name = None
        self.__content = None
        if message == None:
            self.__message = dict()
        else:
            self.__message = message
        self.__set_message()

    def __set_message(self):
        if 'object_device_id' in self.__message.keys():
            self.__object_device_id = self.__message['object_device_id']
        if 'id' in self.__message.keys():
            self.__id = self.__message['id']
        if 'name' in self.__message.keys():
            self.__name = self.__message['name']
        if 'content' in self.__message.keys():
            self.__content = self.__message['content']

    @property
    def device_id(self):
        return self.__object_device_id

    @device_id.setter
    def device_id(self, value):
        self.__object_device_id = value

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, value):
        self.__id = value

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = value

    @property
    def content(self):
        return self.__content

    @content.setter
    def content(self, value):
        self.__content = value
