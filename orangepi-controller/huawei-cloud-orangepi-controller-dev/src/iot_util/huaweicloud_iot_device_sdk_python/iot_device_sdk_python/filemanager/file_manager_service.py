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
import logging
import os

from iot_device_sdk_python.client.request.device_event import DeviceEvent
from iot_device_sdk_python.service.abstract_service import AbstractService
from iot_device_sdk_python.filemanager.url_info import UrlInfo
from iot_device_sdk_python.filemanager.file_manager_listener import FileManagerListener
from iot_device_sdk_python.transport.action_listener import ActionListener
from iot_device_sdk_python.utils.iot_util import get_event_time
from iot_device_sdk_python.utils.iot_util import sha256_hash_from_file


class FileManagerService(AbstractService):
    """
    文件管理器
    """
    _logger = logging.getLogger(__name__)

    def __init__(self):
        super().__init__()
        self._listener: Optional[FileManagerListener] = None
        # file_name -> upload_file_path
        self._upload_file_dict: dict = dict()
        # file_name -> download_file_path
        self._download_file_dict: dict = dict()

    @property
    def upload_file_dict(self):
        return self._upload_file_dict

    @property
    def download_file_dict(self):
        return self._download_file_dict

    def get_listener(self):
        return self._listener

    def set_listener(self, listener: FileManagerListener):
        """
        设置文件管理监听器

        Args:
            listener: 文件管理监听器
        """
        self._listener = listener

    def upload_file(self, file_name: str, file_path: str, file_attributes: Optional[dict] = None):
        if file_name not in self._upload_file_dict.keys():
            self._upload_file_dict[file_name] = file_path
            self.get_upload_file_url(file_name, file_attributes)
        else:
            pass

    def get_upload_file_url(self, file_name: str, file_attributes: Optional[dict] = None,
                            listener: Optional[ActionListener] = None):
        """
        获取文件上传url

        Args:
            file_name:           文件名
            file_attributes:     文件属性
            listener:            发布监听器
        """
        device_event = DeviceEvent()
        device_event.service_id = self.service_id
        device_event.event_type = "get_upload_url"
        device_event.event_time = get_event_time()
        file_attributes_dict = dict()
        if file_attributes is None:
            try:
                file_sha256_hash: str = sha256_hash_from_file(self._upload_file_dict.get(file_name))
                size = os.path.getsize(self._upload_file_dict.get(file_name))
            except Exception as e:
                self._logger.error("sha256 or getsize failed, Exception: %s", str(e))
                raise e
            file_attributes_dict = {"hash_code": file_sha256_hash, "size": size}
        device_event.paras = {"file_name": file_name, "file_attributes": file_attributes_dict}
        self.get_iot_device().get_client().report_event(device_event, listener)

    def download_file(self, file_name: str, file_path: str, file_attributes: Optional[dict] = None):
        if file_name not in self._download_file_dict.keys():
            self._download_file_dict[file_name] = file_path
            self.get_download_file_url(file_name, file_attributes)
        else:
            pass

    def get_download_file_url(self, file_name: str, file_attributes: Optional[dict] = None,
                              listener: Optional[ActionListener] = None):
        """
        获取文件下载url

        Args:
            file_name:           下载文件名
            file_attributes:     文件属性
            listener:            发布监听器
        """
        device_event = DeviceEvent()
        device_event.service_id = self.service_id
        device_event.event_type = "get_download_url"
        device_event.event_time = get_event_time()
        if file_attributes is not None:
            paras: dict = {"file_name": file_name, "file_attributes": file_attributes}
        else:
            paras: dict = {"file_name": file_name}
        device_event.paras = paras
        self.get_iot_device().get_client().report_event(device_event, listener)

    def on_event(self, device_event: DeviceEvent):
        """
        文件服务的事件处理方法

        Args:
            device_event:    事件
        """
        if self._listener is None:
            self._logger.warning("listener in FileManagerService is None, can not process")
            return
        if not isinstance(self._listener, FileManagerListener):
            self._logger.warning("listener is not FileManagerListener, can not process")
            return
        if device_event.event_type == "get_upload_url_response":
            paras: dict = device_event.paras
            url_info = UrlInfo()
            url_info.convert_from_dict(paras)
            self._listener.on_upload_url(url_info)
        elif device_event.event_type == "get_download_url_response":
            paras: dict = device_event.paras
            url_info = UrlInfo()
            url_info.convert_from_dict(paras)
            self._listener.on_download_url(url_info)

    def report_upload_result(self, object_name: str, result_code: int, status_code: Optional[int] = None,
                             status_description: Optional[str] = None, listener: Optional[ActionListener] = None):
        """
        设备上报文件上传结果

        Args:
            object_name:  OBS上传对象名称
            result_code:  设备上传文件状态，0表示上传成功，1表示上传失败
            status_code:  文件上传到OBS返回的状态码
            status_description: 文件上传到OBS时状态的描述
            listener:   发布监听器
        """
        device_event = DeviceEvent()
        device_event.service_id = self.service_id
        device_event.event_type = "upload_result_report"
        device_event.event_time = get_event_time()
        paras: dict = {"object_name": object_name,
                       "result_code": result_code}
        if status_code is not None:
            paras["status_code"] = status_code
        if status_description is not None:
            paras["status_description"] = status_description
        device_event.paras = paras
        self.get_iot_device().get_client().report_event(device_event, listener)

    def report_download_result(self, object_name: str, result_code: int, status_code: Optional[int] = None,
                               status_description: Optional[str] = None, listener: Optional[ActionListener] = None):
        """
        设备上报文件下载结果

        Args:
            object_name:  OBS下载对象名称
            result_code:  设备下载文件状态，0表示上传成功，1表示上传失败
            status_code:  文件下载到OBS返回的状态码
            status_description: 文件下载到OBS时状态的描述
            listener:   发布监听器
        """
        device_event = DeviceEvent()
        device_event.service_id = self.service_id
        device_event.event_type = "download_result_report"
        device_event.event_time = get_event_time()
        paras: dict = {"object_name": object_name,
                       "result_code": result_code}
        if status_code is not None:
            paras["status_code"] = status_code
        if status_description is not None:
            paras["status_description"] = status_description
        device_event.paras = paras
        self.get_iot_device().get_client().report_event(device_event, listener)
