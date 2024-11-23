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
from typing import List
from abc import abstractmethod, ABCMeta

from iot_device_sdk_python.client.request.service_property import ServiceProperty


class PropertyListener(metaclass=ABCMeta):
    """
    属性监听器，用于接收平台下发的属性读写操作
    """

    @abstractmethod
    def on_property_set(self, request_id: str, services: List[ServiceProperty]):
        """
        处理写属性操作

        Args:
            request_id:  请求id
            services:    服务属性列表
        """

    @abstractmethod
    def on_property_get(self, request_id: str, service_id: str):
        """
        处理读属性操作

        Args:
            request_id:  请求id
            service_id:  服务id,可选
        """

