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
import logging
import time
import hmac
import hashlib
import datetime


def get_gmt_timestamp():
    """
    返回当前时间戳，即从格林威治时间1970年01月01日00时00分00秒起至现在的毫秒数

    Returns:
        int: 当前时间戳
    """
    return int(time.time() * 1000)


def get_timestamp():
    return time.strftime("%Y%m%d%H", time.gmtime(time.time()))


def get_event_time():
    """
    获取当前时间，format为 '%Y-%m-%dT%H:%M:%SZ'

    Returns:
        str: 当前时间
    """
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")


def get_client_id(device_id=None, psw_sig_type="0", timestamp=""):
    """
    一机一密的设备clientId由4个部分组成：设备ID、设备身份标识类型（固定值为0）、密码签名类型、时间戳，通过下划线分割
    psw_sig_type为密码签名类型
        '1' 表示检验时间戳，会先校验消息时间戳与平台时间是否一致，在判断密码是否正确。
        '0' 表示不校验时间戳，但也必须带时间戳，但不校验时间是否准确，仅判断密码是否正确。

    Args:
        device_id:  设备id
        psw_sig_type:   密码签名类型
        timestamp: 时间戳
    Returns:
        str: clientId
    """
    if not isinstance(device_id, str):
        raise ValueError("device_id should be a string type")

    return device_id + "_0_" + psw_sig_type + "_" + timestamp


def sha256_hash_from_file(file_path):
    with open(file_path, 'rb') as file:
        sha256obj = hashlib.sha256()
        sha256obj.update(file.read())
        hash_value = sha256obj.hexdigest()
        return hash_value


def sha256_mac(secret, timestamp):
    secret_key = timestamp.encode("utf-8")
    secret = secret.encode("utf-8")
    password = hmac.new(secret_key, secret, digestmod=hashlib.sha256).hexdigest()
    return password


def sha256_mac_salt(secret, salt):
    secret_key = salt.encode("utf-8")
    return hmac.new(secret_key, secret, digestmod=hashlib.sha256).hexdigest()


def get_request_id_from_msg(msg):
    """
    从topic里解析出requestId
    :param msg: 一个RawMessage实例
    :return:    requestId
    """
    topic_list = msg.topic.strip().split("request_id=")
    if len(topic_list) > 1:
        return topic_list[-1]
    else:
        raise ValueError("request_id was not found at message topic")


def get_device_id_from_msg(msg):
    topic_list = msg.topic.strip().split("/")
    device_id_index = topic_list.index("devices") + 1
    if 0 < device_id_index < len(topic_list):
        return topic_list[device_id_index]
    else:
        return None


def str_is_empty(value):
    if value is None:
        return True
    if not isinstance(value, str):
        raise ValueError("Input parameter value is not string")
    return value.strip() == ""


def get_node_id_from_device_id(device_id: str):
    """
    从deviceId解析出nodeId
    :param device_id:   设备id
    :return:    设备物理标识
    """
    try:
        tmp_index = device_id.index("_") + 1
        node_id = device_id[tmp_index:]
    except Exception as e:
        logging.error("get node_id from device_id failed, Exception: %s", str(e))
        return None
    return node_id
