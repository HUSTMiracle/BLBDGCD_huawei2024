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

class CommandRsp:
    """
    命令响应
    """
    def __init__(self):
        self._result_code: int = 0
        self._response_name: str = ""
        self._paras: dict = dict()

    @property
    def result_code(self):
        """
        标识命令的执行结果，0表示成功，其他表示失败。不带默认认为成功。
        """
        return self._result_code

    @result_code.setter
    def result_code(self, value):
        self._result_code = value

    @property
    def response_name(self):
        """
        命令的响应名称，在设备关联的产品模型中定义
        """
        return self._response_name

    @response_name.setter
    def response_name(self, value):
        self._response_name = value

    @property
    def paras(self):
        """
        命令的响应参数，具体字段在设备关联的产品模型中定义
        """
        return self._paras

    @paras.setter
    def paras(self, value):
        self._paras = value

    def to_dict(self):
        """
        将响应内容放到字典中

        Returns:
            dict: 字典形式的响应
        """
        return {"result_code": self._result_code,
                "response_name": self._response_name,
                "paras": self._paras}

    def convert_from_dict(self, json_dict: dict):
        json_name = ["result_code", "response_name", "paras"]
        for key in json_dict.keys():
            if key not in json_name:
                continue
            if key == "result_code":
                self.result_code = json_dict.get(key)
            elif key == "response_name":
                self.response_name = json_dict.get(key)
            elif key == "paras":
                self.paras = json_dict.get(key)
            else:
                pass

    @staticmethod
    def success_code():
        """
        返回成功的结果码

        Returns:
            int: 成功的结果码
        """
        return 0

    @staticmethod
    def fail_code():
        """
        返回失败的结果码

        Returns:
            int: 失败的结果码
        """
        return -1
