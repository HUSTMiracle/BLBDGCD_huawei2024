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
from iot_device_sdk_python.rule.model.actions import Action
from iot_device_sdk_python.rule.model.connditions import Condition
from iot_device_sdk_python.client.request.service_property import ServiceProperty
from iot_device_sdk_python.rule.model.device_info import DeviceInfo
import logging


class ConditionExecute:
    """
        条件校验器
        """
    _logger = logging.getLogger(__name__)

    def __init__(self):
        super().__init__()

    def is_condition_satisfied(self, condition: Condition, properties: List[ServiceProperty]):
        condition_type = condition.type
        if condition_type != "DEVICE_DATA":
            return False
        operator = condition.operator
        device_info: DeviceInfo = condition.device_info
        path_arr: List[str] = device_info.path.split("/")
        if len(path_arr) == 0:
            self._logger.warning("rule condition path is invalid. path: %s", path)
            return False
        service_id_path = path_arr[0]
        property_path = path_arr[len(path_arr) - 1]
        if operator == ">":
            return self.operation_more_than(condition.value, service_id_path, property_path, properties)
        if operator == ">=":
            return self.operation_more_equal(condition.value, service_id_path, property_path, properties)
        if operator == "<":
            return self.operation_less_than(condition.value, service_id_path, property_path, properties)
        if operator == "<=":
            return self.operation_less_equal(condition.value, service_id_path, property_path, properties)
        if operator == "=":
            return self.operation_equal(condition.value, service_id_path, property_path, properties)
        if operator == "between":
            return self.operation_between(condition.value, service_id_path, property_path, properties)
        if operator == "in":
            return self.operation_in(condition.in_values, service_id_path, property_path, properties)
        self._logger.warning("operate is not match. operate: %s", operator)
        return False

    def operation_less_equal(self, value: str, service_id, property_path, properties: List[ServiceProperty]):
        try:
            for prop in properties:
                if service_id == prop.service_id and float(prop.properties.get(property_path)) <= float(value):
                    self._logger.info("match condition for service. service_id: %s, value: %s", service_id, value)
                    return True
        except Exception as e:
            self._logger.warning("operate less equal failed. e: %s", str(e))
        return False

    def operation_less_than(self, value: str, service_id, property_path, properties: List[ServiceProperty]):
        try:
            for prop in properties:
                if service_id == prop.service_id and float(prop.properties.get(property_path)) < float(value):
                    self._logger.info("match condition for service. service_id: %s, value: %s", service_id, value)
                    return True
        except Exception as e:
            self._logger.warning("operate less than failed. e: %s", str(e))
        return False

    def operation_equal(self, value: str, service_id, property_path, properties: List[ServiceProperty]):
        try:
            for prop in properties:
                try:
                    if service_id == prop.service_id and float(prop.properties.get(property_path)) == float(value):
                        self._logger.info("match condition for service. service_id: %s, value: %s", service_id, value)
                        return True
                except ValueError:
                    if service_id == prop.service_id and prop.properties.get(property_path) == value:
                        self._logger.info("match condition for service. service_id: %s, value: %s", service_id, value)
                        return True
        except Exception as e:
            self._logger.warning("operate equal failed. e: %s", str(e))
        return False

    def operation_more_equal(self, value: str, service_id, property_path, properties: List[ServiceProperty]):
        try:
            for prop in properties:
                if service_id == prop.service_id and float(prop.properties.get(property_path)) >= float(value):
                    self._logger.info("match condition for service. service_id: %s, value: %s", service_id, value)
                    return True
        except Exception as e:
            self._logger.warning("operate more equal failed. e: %s", str(e))
        return False

    def operation_more_than(self, value: str, service_id, property_path, properties: List[ServiceProperty]):
        try:
            for prop in properties:
                if service_id == prop.service_id and float(prop.properties.get(property_path)) > float(value):
                    self._logger.info("match condition for service. service_id: %s, value: %s", service_id, value)
                    return True
        except Exception as e:
            self._logger.warning("operate more than failed. e: %s", str(e))
        return False

    def operation_between(self, value: str, service_id, property_path, properties: List[ServiceProperty]) -> bool:
        try:
            value_list: List[float] = list(map(float, value.split(",")))
            if len(value_list) != 2:
                self._logger.warning("rule condition value is invalid. value: %s", value)
                return False
            for prop in properties:
                if service_id == prop.service_id and value_list[0] <= float(prop.properties.get(property_path)) <= value_list[1]:
                    self._logger.info("match condition for service. service_id: %s, value: %s", service_id, value)
                    return True
        except Exception as e:
            self._logger.warning("operate between failed. e: %s", str(e))
        return False

    def operation_in(self, value: List[str], service_id, property_path, properties: List[ServiceProperty]) -> bool:
        try:
            value_list: List[float] = list(map(float, value))
            for prop in properties:
                if service_id == prop.service_id and float(prop.properties.get(property_path)) in value_list:
                    self._logger.info("match condition for service. service_id: %s, value: %s", service_id, value)
                    return True
        except Exception as e:
            self._logger.warning("operate operation in failed. e: %s", str(e))
        return False
