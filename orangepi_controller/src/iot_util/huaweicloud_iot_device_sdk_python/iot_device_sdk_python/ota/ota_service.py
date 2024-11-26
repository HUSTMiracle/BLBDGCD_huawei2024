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
from typing import Optional
import threading
import logging

from iot_device_sdk_python.client.request.device_event import DeviceEvent
from iot_device_sdk_python.ota.ota_query_version import OTAQueryVersion
from iot_device_sdk_python.transport.action_listener import ActionListener
from iot_device_sdk_python.utils.iot_util import get_event_time
from iot_device_sdk_python.ota.ota_listener import OTAListener
from iot_device_sdk_python.ota.ota_package_info import OTAPackageInfo
from iot_device_sdk_python.ota.ota_package_info_v2 import OTAPackageInfoV2
from iot_device_sdk_python.service.abstract_service import AbstractService


class OTAService(AbstractService):
    """
    OTA服务类，提供设备升级相关接口
    """
    # 升级上报的错误码，用户也可以扩展自己的错误码
    OTA_CODE_SUCCESS = 0  # 成功
    OTA_CODE_BUSY = 1  # 设备使用中
    OTA_CODE_SIGNAL_BAD = 2  # 信号质量差
    OTA_CODE_NO_NEED = 3  # 已经是最新版本
    OTA_CODE_LOW_POWER = 4  # 电量不足
    OTA_CODE_LOW_SPACE = 5  # 剩余空间不足
    OTA_CODE_DOWNLOAD_TIMEOUT = 6  # 下载超时
    OTA_CODE_CHECK_FAIL = 7  # 升级包校验失败
    OTA_CODE_UNKNOWN_TYPE = 8  # 升级包类型不支持
    OTA_CODE_LOW_MEMORY = 9  # 内存不足
    OTA_CODE_INSTALL_FAIL = 10  # 安装升级包失败
    OTA_CODE_INNER_ERROR = 255  # 内部异常

    _logger = logging.getLogger(__name__)

    def __init__(self):
        super().__init__()
        self._ota_listener: Optional[OTAListener] = None

    def set_ota_listener(self, ota_listener: OTAListener):
        """
        设置OTA监听器

        Args:
            ota_listener:    OTA监听器
        """
        self._ota_listener = ota_listener

    def on_event(self, device_event: DeviceEvent):
        """
        接收OTA事件处理

        Args:
            device_event: 设备事件
        """
        if self._ota_listener is None:
            self._logger.warning("ota listener in OTAService is None, can not process")
            return
        if not isinstance(self._ota_listener, OTAListener):
            self._logger.warning("ota listener is not OTAListener, can not process")
            return

        if device_event.event_type == "version_query":
            # 查询版本
            info: dict = device_event.paras
            queryInfo = OTAQueryVersion()
            if info is not None:
                queryInfo.convert_from_dict(info)
            self._ota_listener.on_query_version(queryInfo)
        elif device_event.event_type == "firmware_upgrade" or device_event.event_type == "software_upgrade":
            # 版本升级
            pkg: dict = device_event.paras
            ota_pkg = OTAPackageInfo()
            ota_pkg.convert_from_dict(pkg)
            # 因为版本升级需要下载包，所以要在另一个线程中调用one_new_package方法
            threading.Thread(target=self._ota_listener.on_receive_package_info(ota_pkg)).start()
        elif device_event.event_type == "firmware_upgrade_v2" or device_event.event_type == "software_upgrade_v2":
            # 版本升级
            pkg: dict = device_event.paras
            ota_pkg = OTAPackageInfoV2()
            ota_pkg.convert_from_dict(pkg)
            # 因为版本升级需要下载包，所以要在另一个线程中调用one_new_package方法
            threading.Thread(target=self._ota_listener.on_receive_package_info(ota_pkg)).start()

    def report_version(self, fw_version: str = None, sw_version: str = None, listener: Optional[ActionListener] = None):
        """
        上报固件版本信息

        Args:
            fw_version:     固件版本
            sw_version:     软件版本
            listener:       发布监听器
        """
        paras = dict()
        if fw_version is not None:
            paras["fw_version"] = fw_version
        if sw_version is not None:
            paras["sw_version"] = sw_version

        device_event = DeviceEvent()
        device_event.event_type = "version_report"
        device_event.paras = paras
        device_event.service_id = "$ota"
        device_event.event_time = get_event_time()
        self.get_iot_device().get_client().report_event(device_event, listener)

    def report_ota_status(self, result_code: int, version: str, progress: int = None, description: str = None,
                          listener: Optional[ActionListener] = None) -> object:
        """
        上报升级状态

        Args:
            result_code:     升级结果
            progress:        升级进度0-100
            version:         当前版本
            description:     具体失败的原因，可选参数
            listener:        发布监听器
        """
        paras = {"result_code": result_code, "version": version}
        if progress is not None:
            paras["progress"] = progress
        if description is not None:
            paras["description"] = description

        device_event = DeviceEvent()
        device_event.event_type = "upgrade_progress_report"
        device_event.paras = paras
        device_event.service_id = "$ota"
        device_event.event_time = get_event_time()

        self.get_iot_device().get_client().report_event(device_event, listener)
