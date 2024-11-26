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
from unittest import TestCase, mock
from copy import deepcopy

from paho.mqtt.client import MQTTMessage

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.gateway.abstract_gateway import AbstractGateway
from iot_device_sdk_python.gateway.gtw_operate_sub_device_listener import GtwOperateSubDeviceListener
from iot_device_sdk_python.gateway.requests.add_sub_device_failed_reason import AddSubDeviceFailedReason
from iot_device_sdk_python.gateway.requests.added_sub_device_info_rsp import AddedSubDeviceInfoRsp
from iot_device_sdk_python.gateway.requests.del_sub_device_failed_reason import DelSubDeviceFailedReason
from iot_device_sdk_python.gateway.requests.device_info import DeviceInfo
from iot_device_sdk_python.gateway.requests.gtw_add_sub_device_rsp import GtwAddSubDeviceRsp
from iot_device_sdk_python.gateway.requests.gtw_del_sub_device_rsp import GtwDelSubDeviceRsp
from iot_device_sdk_python.gateway.requests.sub_devices_info import SubDevicesInfo
from iot_device_sdk_python.gateway.sub_devices_persistence import SubDevicesPersistence
from iot_gateway_demo.sub_dev_info import SubDevInfo


class SubDevicesDictPersistence(SubDevicesPersistence):
    """
    将子设备信息保存到json文件。用户可以自己实现SubDevicesPersistence接口来进行替换
    """
    def __init__(self, del_device: list):
        self.del_device = del_device
        self._sub_dev_info_cache = SubDevInfo()
        self.conf_dict = dict()
        init_sub_dev_info = SubDevInfo()
        init_sub_dev_info.version = 0
        init_sub_dev_info.sub_devices = {"1": DeviceInfo().to_dict()}
        self.conf_dict = init_sub_dev_info.to_dict()

    def _read_conf_dict(self):
        load_dict = deepcopy(self.conf_dict)
        self._sub_dev_info_cache.version = load_dict["version"]
        self._sub_dev_info_cache.sub_devices = load_dict["sub_devices"]

    def _write_conf_dict(self):
        self.conf_dict = deepcopy(self._sub_dev_info_cache.to_dict())

    def get_sub_device(self, node_id: str):
        if node_id in self._sub_dev_info_cache.sub_devices.keys():
            device: dict = self._sub_dev_info_cache.sub_devices.get(node_id)
            device_info = DeviceInfo()
            device_info.convert_from_dict(device)
            return device_info
        else:
            return None

    def add_sub_devices(self, sub_devices_info: SubDevicesInfo):
        if sub_devices_info is None:
            return -1

        if 0 < sub_devices_info.version <= self._sub_dev_info_cache.version:
            return -1

        if self.add_sub_device_to_dict(sub_devices_info) != 0:
            return -1
        return 0

    def delete_sub_devices(self, sub_devices_info: SubDevicesInfo):
        if sub_devices_info is None:
            return -1

        self.del_device.append(sub_devices_info.devices[0].device_id)

        if 0 < sub_devices_info.version <= self._sub_dev_info_cache.version:
            return -1

        if self.rmv_sub_device_to_dict(sub_devices_info) != 0:
            return -1
        return 0

    def get_version(self):
        return self._sub_dev_info_cache.version

    def add_sub_device_to_dict(self, sub_devices_info: SubDevicesInfo):
        self._read_conf_dict()
        try:
            for device in sub_devices_info.devices:
                device_node_id = device.node_id
                self._sub_dev_info_cache.sub_devices[device_node_id] = device.to_dict()
            self._sub_dev_info_cache.version = sub_devices_info.version
        except Exception:
            return -1
        self._write_conf_dict()
        return 0

    def rmv_sub_device_to_dict(self, sub_devices_info: SubDevicesInfo):
        self._read_conf_dict()
        try:
            for device in sub_devices_info.devices:
                device_node_id = device.node_id
                if device_node_id not in self._sub_dev_info_cache.sub_devices.keys():
                    # 若pop一个不存在的对象会报错
                    pass
                else:
                    self._sub_dev_info_cache.sub_devices.pop(device_node_id)
            self._sub_dev_info_cache.version = sub_devices_info.version
        except Exception:
            return -1
        self._write_conf_dict()
        return 0


class GtwOpSubDeviceTestListener(GtwOperateSubDeviceListener):
    """
    网关新增/删除子设备请求响应监听器
    此例子将响应的内容打印出来，并向平台请求同步子设备信息
    """
    def __init__(self, result_dict: dict):
        self.result_dict = result_dict

    def on_add_sub_device_rsp(self, gtw_add_sub_device_rsp: GtwAddSubDeviceRsp, event_id: str):
        """
        处理网关增加子设备返回结果
        :param gtw_add_sub_device_rsp:  网关增加子设备响应
        :param event_id:    事件id
        """
        # success
        for rsp in gtw_add_sub_device_rsp.successful_devices:
            rsp: AddedSubDeviceInfoRsp
            self.result_dict["add_success_rsp_device_id"] = rsp.device_id
            return
        # failed
        for rsp in gtw_add_sub_device_rsp.add_sub_device_failed_reasons:
            rsp: AddSubDeviceFailedReason
            self.result_dict["add_fail_rsp"] = rsp
            return

    def on_del_sub_device_rsp(self, gtw_del_sub_device_rsp: GtwDelSubDeviceRsp, event_id: str):
        """
        处理网关删除子设备返回结果
        :param gtw_del_sub_device_rsp:  网关删除子设备响应
        :param event_id:    事件id
        """
        # success
        for rsp in gtw_del_sub_device_rsp.successful_devices:
            rsp: str
            self.result_dict["del_success_rsp_device_id"] = rsp
            return
        # failed
        for rsp in gtw_del_sub_device_rsp.failed_devices:
            rsp: DelSubDeviceFailedReason
            self.result_dict["del_fail_rsp"] = rsp
            return


class TestGatewayMethod(TestCase):

    @mock.patch("paho.mqtt.client.Client.is_connected")
    @mock.patch("paho.mqtt.client.Client.loop_forever")
    @mock.patch("paho.mqtt.client.Client.connect")
    def setUp(self, connect, loop_forever, is_connected):
        # mock
        connect.return_value = 0
        loop_forever.return_value = 0
        is_connected.return_value = True

        # setup
        server_uri = "access address"
        port = 8883
        self.device_id = "productId_nodeId"
        secret = "12345678"
        iot_ca_cert_path = "./resources/root.pem"

        connect_auth_info = ConnectAuthInfo()
        connect_auth_info.server_uri = server_uri
        connect_auth_info.port = port
        connect_auth_info.id = self.device_id
        connect_auth_info.secret = secret
        connect_auth_info.iot_cert_path = iot_ca_cert_path
        connect_auth_info.auth_type = ConnectAuthInfo.SECRET_AUTH
        connect_auth_info.bs_mode = ConnectAuthInfo.BS_MODE_DIRECT_CONNECT

        client_conf = ClientConf(connect_auth_info)

        self.del_device = list()
        self.persistence = SubDevicesDictPersistence(self.del_device)
        self.device = AbstractGateway(self.persistence, client_conf)
        self.result_dict = dict()
        self.device.set_gtw_operate_sub_device_listener(GtwOpSubDeviceTestListener(self.result_dict))

        self.device.connect()
        return

    def test_gtw_receive_add_device(self):
        # test
        _get_connection = getattr(self.device.get_client(), "_get_connection")
        _connection = _get_connection()
        _get_paho_client = getattr(_connection, "_get_paho_client")
        _paho_client = _get_paho_client()
        _handle_on_message = getattr(_paho_client, "_handle_on_message")
        message = MQTTMessage()
        message.topic = b'$oc/devices/' \
                        + self.device_id.encode(encoding="utf-8") \
                        + b'/sys/events/down'
        message.payload = b'{"object_device_id": "' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'","command_name": "TestCommandName","service_id": "WaterMeter","paras": {"value": "1"}}'
        message.payload = b'{"object_device_id": "' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'","services": [{"service_id": "$sub_device_manager",' \
                            b'"event_type": "add_sub_device_notify","event_time": "20151212T121212Z",' \
                          + b'"paras": {"devices": [{"parent_device_id": "c6b39067b0325db34663d3ef421a42f6_12345678",' \
                            b'"node_id": "subdevice11","device_id": "2bb4ddba-fb56-4566-8577-063ad2f5a6cc",' \
                            b'"name": "subDevice11","description": null,"manufacturer_id": "ofo","model": "twx2",' \
                            b'"product_id": "c6b39067b0325db34663d3ef421a42f6","fw_version": null,"sw_version": null,' \
                            b'"status": "ONLINE"}],"version": 1}}]}'
        _handle_on_message(message)

        # assert
        self.assertEqual(self.persistence.conf_dict["sub_devices"]["subdevice11"]["device_id"],
                         "2bb4ddba-fb56-4566-8577-063ad2f5a6cc")
        self.assertEqual(self.persistence.conf_dict["sub_devices"]["subdevice11"]["product_id"],
                         "c6b39067b0325db34663d3ef421a42f6")
        return

    def test_gtw_receive_del_device(self):
        # test
        _get_connection = getattr(self.device.get_client(), "_get_connection")
        _connection = _get_connection()
        _get_paho_client = getattr(_connection, "_get_paho_client")
        _paho_client = _get_paho_client()
        _handle_on_message = getattr(_paho_client, "_handle_on_message")
        message = MQTTMessage()
        message.topic = b'$oc/devices/' \
                        + self.device_id.encode(encoding="utf-8") \
                        + b'/sys/events/down'
        message.payload = b'{"object_device_id": "' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'","command_name": "TestCommandName","service_id": "WaterMeter","paras": {"value": "1"}}'
        message.payload = b'{"object_device_id": "' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'","services": [{"service_id": "$sub_device_manager",' \
                            b'"event_type": "delete_sub_device_notify","event_time": "20151212T121212Z",' \
                          + b'"paras": {"devices": [{"parent_device_id": "c6b39067b0325db34663d3ef421a42f6_12345678",' \
                            b'"node_id": "subdevice11","device_id": "2bb4ddba-fb56-4566-8577-063ad2f5a6cc"}],' \
                            b'"version": 1}}]}'
        _handle_on_message(message)

        # assert
        self.assertEqual(self.del_device[0], "2bb4ddba-fb56-4566-8577-063ad2f5a6cc")
        return

    def test_gtw_receive_op_add_device_resp(self):
        # test
        _get_connection = getattr(self.device.get_client(), "_get_connection")
        _connection = _get_connection()
        _get_paho_client = getattr(_connection, "_get_paho_client")
        _paho_client = _get_paho_client()
        _handle_on_message = getattr(_paho_client, "_handle_on_message")
        message = MQTTMessage()
        message.topic = b'$oc/devices/' \
                        + self.device_id.encode(encoding="utf-8") \
                        + b'/sys/events/down'
        message.payload = b'{"object_device_id": "' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'","command_name": "TestCommandName","service_id": "WaterMeter","paras": {"value": "1"}}'
        message.payload = b'{"object_device_id": "' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'","services": [{"service_id": "$sub_device_manager",' \
                            b'"event_type": "add_sub_device_response","event_time": "20151212T121212Z",' \
                          + b'"paras": {"successful_devices": [{' \
                            b'"device_id": "c6b39067b0325db34663d3ef421a42f6_subdevice11",' \
                            b'"name": "subdevice11","node_id": "subdevice11",' \
                            b'"product_id": "c6b39067b0325db34663d3ef421a42f6","description": "subdevice11",' \
                            b'"manufacturer_id": "ofo","model": "twx2","fw_version": null,"sw_version": null,' \
                            b'"status": "ONLINE","extension_info" : null,"parent_device_id" : null}],' \
                            b'"failed_devices": []}}]}'
        _handle_on_message(message)

        # assert
        self.assertEqual(self.result_dict["add_success_rsp_device_id"], "c6b39067b0325db34663d3ef421a42f6_subdevice11")
        return

    def test_gtw_receive_op_del_device_resp(self):
        # test
        _get_connection = getattr(self.device.get_client(), "_get_connection")
        _connection = _get_connection()
        _get_paho_client = getattr(_connection, "_get_paho_client")
        _paho_client = _get_paho_client()
        _handle_on_message = getattr(_paho_client, "_handle_on_message")
        message = MQTTMessage()
        message.topic = b'$oc/devices/' \
                        + self.device_id.encode(encoding="utf-8") \
                        + b'/sys/events/down'
        message.payload = b'{"object_device_id": "' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'","command_name": "TestCommandName","service_id": "WaterMeter","paras": {"value": "1"}}'
        message.payload = b'{"object_device_id": "' \
                          + self.device_id.encode(encoding="utf-8") \
                          + b'","services": [{"service_id": "$sub_device_manager",' \
                            b'"event_type": "delete_sub_device_response","event_time": "20151212T121212Z",' \
                          + b'"paras": {"successful_devices": ["c6b39067b0325db34663d3ef421a42f6_subdevice11"],' \
                            b'"failed_devices": []}}]}'
        _handle_on_message(message)

        # assert
        self.assertEqual(self.result_dict["del_success_rsp_device_id"], "c6b39067b0325db34663d3ef421a42f6_subdevice11")
        return

