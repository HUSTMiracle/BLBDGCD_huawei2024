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

class Property:
    """
    物模型的属性
    """
    def __init__(self, val, field_name: str = "", prop_name: str = "", writeable=True):
        self._val = val
        self._field_name: str = field_name
        self._prop_name: str = prop_name
        self._writeable: bool = writeable

    @property
    def val(self):
        """
        属性的值
        """
        return self._val

    @val.setter
    def val(self, value):
        self._val = value

    @property
    def field_name(self):
        """
        属性对应的变量的名称
        """
        return self._field_name

    @field_name.setter
    def field_name(self, value):
        self._field_name = value

    @property
    def prop_name(self):
        """
        属性的名称，需要和产品模型定义的一致
        """
        return self._prop_name

    @prop_name.setter
    def prop_name(self, value):
        self._prop_name = value

    @property
    def writeable(self):
        """
        属性是否可写
        """
        return self._writeable

    @writeable.setter
    def writeable(self, value):
        self._writeable = value

