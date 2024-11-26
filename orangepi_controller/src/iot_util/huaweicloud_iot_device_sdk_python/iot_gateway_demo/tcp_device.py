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
from typing import Optional
import asyncio

from tornado import ioloop, gen
from tornado.tcpclient import TCPClient
from tornado.iostream import IOStream, StreamClosedError


class TcpDevice:
    """
    一个tcp客户端，仅用于测试
    """
    def __init__(self, host, port):
        self._logger = logging.getLogger(TcpDevice.__name__)
        self._host = host
        self._port = port
        self.stream: Optional[IOStream] = None
        self._device_id = "you sud device id"    # 子设备id，由用户填写, 在此示例中必须在中间包含一个下划线，如abc_efg

    @gen.coroutine
    def start(self):
        self.stream = yield TCPClient().connect(self._host, self._port)
        try:
            while True:
                yield self.send_message()
                yield self.receive_message()
        except StreamClosedError as e:
            print(e)

    @gen.coroutine
    def send_message(self):
        msg = input("input:")
        msg = self._device_id + "|" + msg
        bytes_msg = bytes(msg.encode("utf-8"))
        yield self.stream.write(bytes_msg)

    @gen.coroutine
    def receive_message(self):
        try:
            msg = yield self.stream.read_bytes(1024, partial=True)
            str_msg = msg.decode("utf-8")
            print("receive message: ", str_msg)
        except StreamClosedError as e:
            print(e)
            pass


if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    td = TcpDevice("localhost", 14322)
    td.start()
    ioloop.IOLoop.current().start()

