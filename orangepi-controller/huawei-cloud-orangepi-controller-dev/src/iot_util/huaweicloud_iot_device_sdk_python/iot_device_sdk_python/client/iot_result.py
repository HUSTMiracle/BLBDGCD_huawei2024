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

class IotResult:
    """
    处理结果
    """
    def __init__(self, result_code: int, result_desc: str):
        self._result_code: int = result_code
        self._result_desc: str = result_desc

    @property
    def result_code(self):
        """
        结果码，0表示成功，其他为失败
        """
        return self._result_code

    @result_code.setter
    def result_code(self, value):
        self._result_code = value

    @property
    def result_desc(self):
        """
        结果描述
        """
        return self._result_desc

    @result_desc.setter
    def result_desc(self, value):
        self._result_desc = value

    def to_dict(self):
        return {"result_code": self._result_code, "result_desc": self._result_desc}


SUCCESS = IotResult(0, "Success")
FAIL = IotResult(1, "Fail")
TIMEOUT = IotResult(2, "Timeout")

