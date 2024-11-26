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

import logging

from iot_device_sdk_python.gateway.gtw_operate_sub_device_listener import GtwOperateSubDeviceListener, \
    GtwAddSubDeviceRsp, GtwDelSubDeviceRsp
from iot_device_sdk_python.gateway.requests.add_sub_device_failed_reason import AddSubDeviceFailedReason
from iot_device_sdk_python.gateway.requests.added_sub_device_info_rsp import AddedSubDeviceInfoRsp
from iot_device_sdk_python.gateway.requests.del_sub_device_failed_reason import DelSubDeviceFailedReason
from iot_device_sdk_python.gateway.abstract_gateway import AbstractGateway


class GtwOpSubDeviceListener(GtwOperateSubDeviceListener):
    """
    网关新增/删除子设备请求响应监听器
    此例子将响应的内容打印出来，并向平台请求同步子设备信息
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, gateway: AbstractGateway):
        self._gateway = gateway

    def on_add_sub_device_rsp(self, gtw_add_sub_device_rsp: GtwAddSubDeviceRsp, event_id: str):
        """
        处理网关增加子设备返回结果
        :param gtw_add_sub_device_rsp:  网关增加子设备响应
        :param event_id:    事件id
        """
        # success
        for rsp in gtw_add_sub_device_rsp.successful_devices:
            rsp: AddedSubDeviceInfoRsp
            self._logger.info("add sub device success: device_id: " + rsp.device_id)
        # failed
        for rsp in gtw_add_sub_device_rsp.add_sub_device_failed_reasons:
            rsp: AddSubDeviceFailedReason
            self._logger.info("add sub device failed: node_id: " + rsp.node_id
                              + " error_code: " + rsp.error_code
                              + " error_msg: " + rsp.error_msg)
        # 更新一次子设备信息
        self._gateway.sync_sub_devices()
        pass

    def on_del_sub_device_rsp(self, gtw_del_sub_device_rsp: GtwDelSubDeviceRsp, event_id: str):
        """
        处理网关删除子设备返回结果
        :param gtw_del_sub_device_rsp:  网关删除子设备响应
        :param event_id:    事件id
        """
        # success
        for rsp in gtw_del_sub_device_rsp.successful_devices:
            rsp: str
            self._logger.info("del sub device success: device_id: " + rsp)
        # failed
        for rsp in gtw_del_sub_device_rsp.failed_devices:
            rsp: DelSubDeviceFailedReason
            self._logger.info("del sub device failed: device_id: " + rsp.device_id
                              + " error_code: " + rsp.error_code
                              + " error_msg: " + rsp.error_msg)
        # 更新一次子设备信息
        self._gateway.sync_sub_devices()
        pass

