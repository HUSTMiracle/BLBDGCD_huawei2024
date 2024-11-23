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

'''
定义服务的属性
'''


class ServicesProperties:
    def __init__(self):
        self.__services_properties = list()

    def add_service_property(self, service_id, property, value):
        service_property_dict = {
            "service_id": service_id, "properties": {property: value}}
        self.__services_properties.append(service_property_dict)

    @property
    def service_property(self):
        return self.__services_properties
