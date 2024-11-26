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
from abc import abstractmethod, ABCMeta
from iot_device_sdk_python.rule.model.actions import Action


class ActionHandler(metaclass=ABCMeta):

    @abstractmethod
    def handle_rule_action(self, action: List[Action]):
        """
        自定义规则触发器，用于客户自定义触发规则
        Args:
            action:  端侧规则动作
        """
