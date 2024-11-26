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

from iot_device_sdk_python.ota.ota_package_info import OTAPackageInfo
from iot_device_sdk_python.ota.ota_package_info_v2 import OTAPackageInfoV2
from typing import Union

from iot_device_sdk_python.ota.ota_query_version import OTAQueryVersion


class OTAListener(metaclass=ABCMeta):
    """
    OTA监听器
    """

    @abstractmethod
    def on_query_version(self, queryInfo: OTAQueryVersion):
        """
        接收查询版本通知
        """

    @abstractmethod
    def on_receive_package_info(self, pkg: Union[OTAPackageInfo, OTAPackageInfoV2]):
        """
        接收版本包信息，下载包并安装

        Args:
            pkg:     新版本包信息
        """

