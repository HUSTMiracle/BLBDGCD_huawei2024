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
from abc import abstractmethod, ABCMeta


class Connection(metaclass=ABCMeta):
    """
    Iot连接，代表设备和平台之间的一个连接
    """

    @abstractmethod
    def connect(self):
        """
        建立连接

        Returns:
            int: 建立连接结果，0表示成功，其他表示失败
        """

    @abstractmethod
    def publish_message(self, message, listener):
        """
        发布消息    （参数是否需要 message 和 listener）

        Args:
            message:     原始数据
            listener:    监听器，可以为None
        """

    @abstractmethod
    def close(self):
        """
        关闭连接
        """

    @abstractmethod
    def is_connected(self):
        """
        是否连接中
        """

    @abstractmethod
    def add_connect_listener(self, connect_listener):
        """
        设置链路监听器

        Args:
            connect_listener:    链路监听器
        """

    @abstractmethod
    def set_connect_action_listener(self, connect_action_listener):
        """
        设置连接动作监听器

        Args:
            connect_action_listener: 连接动作监听器
        """

    @abstractmethod
    def subscribe_topic(self, topic: str, qos: int):
        """
        订阅自定义topic

        Args:
            topic: 自定义的topic
            qos:   qos
        """
