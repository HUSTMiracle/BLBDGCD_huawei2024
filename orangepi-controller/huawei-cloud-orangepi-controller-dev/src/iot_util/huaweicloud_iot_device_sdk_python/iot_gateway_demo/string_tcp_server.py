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

import uuid
import traceback
import asyncio

from tornado import ioloop, gen
from tornado.tcpserver import TCPServer
from tornado.iostream import IOStream, StreamClosedError

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.client.request.service_property import ServiceProperty
from iot_gateway_demo.simple_gateway import SimpleGateway
from iot_gateway_demo.sub_devices_file_persistence import SubDevicesFilePersistence
from iot_gateway_demo.gtw_op_sub_device_listener import GtwOpSubDeviceListener
from iot_device_sdk_python.client.request.device_message import DeviceMessage
from iot_device_sdk_python.utils.iot_util import get_node_id_from_device_id
from iot_device_sdk_python.gateway.requests.added_sub_device_info import AddedSubDeviceInfo

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(threadName)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")


class SimpleHandler:
    _logger = logging.getLogger(__name__)

    def __init__(self):
        # 替换为真正的接入地址、device id和密钥，参考readme_CN.md中“上传产品模型并注册设备”
        server_uri = "access address"
        port = 8883
        device_id = "your device id"  # 填入从云平台获取的设备id
        secret = "your device secret"  # 填入从云平台获取的设备密钥
        iot_ca_cert_path = "./resources/root.pem"

        connect_auth_info = ConnectAuthInfo()
        connect_auth_info.server_uri = server_uri
        connect_auth_info.port = port
        connect_auth_info.id = device_id
        connect_auth_info.secret = secret
        connect_auth_info.iot_cert_path = iot_ca_cert_path
        connect_auth_info.bs_mode = ConnectAuthInfo.BS_MODE_DIRECT_CONNECT

        client_conf = ClientConf(connect_auth_info)

        self._gateway = SimpleGateway(SubDevicesFilePersistence(), client_conf)
        self._gateway.create_by_secret(server_uri=server_uri,
                                       port=port,
                                       device_id=device_id,
                                       secret=secret,
                                       iot_cert_file=iot_ca_cert_path)
        # 设置网关新增/删除子设备请求响应监听器
        self._gateway.set_gtw_operate_sub_device_listener(GtwOpSubDeviceListener(self._gateway))

        if self._gateway.connect() != 0:
            raise RuntimeError("gateway init failed")
        # 连接成功后向平台请求同步子设备信息
        self._gateway.sync_sub_devices()

    def report_message(self, stream: IOStream, msg: str):
        """
        网关收到来自子设备的消息
        :param stream:  IOStream
        :param msg:     消息
        """
        device_id, content = msg.split("|")[:2]

        node_id = get_node_id_from_device_id(device_id)
        try:
            if self._gateway.get_session(node_id) is None:
                # 这是首条消息，创建session
                session = self._gateway.create_session(node_id, stream)
                if session is None:
                    # 创建session失败，说明此子设备并没有在平台上注册，这里演示网关主动新增子设备
                    self._logger.info("create session failed, register device: " + device_id)
                    uid = str(uuid.uuid1())
                    device_info = AddedSubDeviceInfo()
                    device_info.parent_device_id = self._gateway.get_device_id()
                    device_info.node_id = node_id
                    device_info.product_id = "668619463030681a7de586d6"     # 产品id，由用户自行修改
                    device_info.description = "new device at node {}".format(node_id)   # 必填
                    device_info.device_id = device_id
                    device_info.name = node_id
                    infos = [device_info]
                    self._gateway.gtw_add_sub_device(added_sub_device_infos=infos, event_id=uid)
                    return "sub device: %s add" % device_id
                else:
                    # 创建session成功，通知平台更新子设备状态为ONLINE
                    self._logger.info("ready to go online, device_id: " + device_id)
                    self._gateway.report_sub_device_status(device_id, "ONLINE")
                    return "sub device: %s go online" % device_id
            else:
                """ 
                根据子设备发来的消息来做不同的操作。
                如果子设备发来的content为 gtwdel ，那么网关会给平台发送一个删除子设备的请求
                如果子设备发来的content为其它内容，网关会发送一条信息给平台
                """
                if content == "gtwdel":
                    # 这里演示网关主动删除子设备请求
                    uid = str(uuid.uuid1())
                    del_sub_device = device_id
                    del_sub_devices = [del_sub_device]
                    self._gateway.gtw_del_sub_device(del_sub_devices=del_sub_devices, event_id=uid)
                    return "sub device: %s delete" % device_id
                else:
                    # 网关收到子设备上行数据，可以以消息或属性上报转发到平台
                    # 这里演示上报消息
                    device_message = DeviceMessage()
                    device_message.content = "Hello Huawei"
                    device_message.device_id = device_id
                    self._gateway.report_sub_device_message(device_message)
                    # 这里演示上报属性
                    service_property = ServiceProperty()
                    # 属性值暂且写死，实际中应该根据子设备上报的进行组装
                    service_property.service_id = "smokeDetector"
                    service_property.properties = {"alarm": 10, "smokeConcentration": 36, "temperature": 64,
                                                   "humidity": 32}
                    self._gateway.report_sub_device_properties(device_id=device_id,
                                                               services=[service_property],
                                                               listener=None)
                    return "sub device: %s report message and properties" % device_id
        except Exception as e:
            self._logger.error("server process failed, traceback: " + traceback.format_exc())


class StringTcpServer(TCPServer):
    def __init__(self):
        super(StringTcpServer, self).__init__()
        self.handler = SimpleHandler()

    @gen.coroutine
    def handle_stream(self, stream: IOStream, address: tuple):
        try:
            while True:
                msg = yield stream.read_bytes(1024, partial=True)
                str_msg = msg.decode("utf-8")
                str_rsp = self.handler.report_message(stream, str_msg)
                rsp = bytes(str_rsp.encode("utf-8"))
                yield stream.write(rsp)
        except StreamClosedError as e:
            print(e)
        pass


if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sts = StringTcpServer()
    sts.listen(14322)
    ioloop.IOLoop.current().start()

