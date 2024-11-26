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

import utime
import hmac
from hashlib import sha256


# 获得时间戳
def get_timestamp():
    """
    :return:时间戳：为设备连接平台时的UTC时间，格式为YYYYMMDDHH，如UTC 时间2020/04/26 19:56:20 则应表示为2020032619。
    """
    t = utime.localtime(utime.time())
    return "{:>04}{:>02}{:>02}{:>02}".format(*t[0:4])


# 获取客户ID
def get_client_id(device_id=None, psw_sig_type=0):
    """
    一机一密的设备clientId由4个部分组成：设备ID、设备身份标识类型、密码签名类型、时间戳。通过下划线“_”分隔。
    :param deviceId: 注册时的设备ID，可在配置文件设置，即config.py中的deviceId
    :param device_id_type: 设备身份标识类型固定值为0
    :param Psw_sig_type:  密码签名类型：长度1字节，当前支持2种类型：
                          “0”代表HMACSHA256不校验时间戳。
                          “1”代表HMACSHA256校验时间戳。
    :param time_stamp: 时间戳
    """
    if not isinstance(device_id, str):
        raise ValueError('device_id should be a string type')

    return device_id + '_0_' + str(psw_sig_type) + '_' + get_timestamp()


def get_password(secret):
    """
    对 secret 进行加密
    :param secret: 返回的password的值为使用“HMACSHA256”算法以时间戳为秘钥，对secret进行加密后的值。secret为注册设备时平台返回的secret。
    """
    secret_key = get_timestamp().encode('utf-8')  # 秘钥
    secret = secret.encode('utf-8')  # 加密数据
    password = hmac.new(secret_key, secret, digestmod=sha256).hexdigest()
    return password


def get_request_id_from_topic(topic):
    """
    :param msg: 一般为平台下发的消息，包含 topic 和 payload
    :return:  request_id
    """
    topic_list = topic.strip().split('request_id=')
    if len(topic_list) > 1:
        return topic_list[-1]  # request_i紧接着'request_id='在topic的最后部分
    else:
        raise ValueError('request_id was not found at message topic')


def get_device_id_from_topic(topic):
    '''
    查找 device id
    :param msg:一般为平台下发的消息，包含 topic 和 payload
    '''
    topic_list = topic.strip().split('/')
    device_id_index = topic_list.index('devices') + 1
    if 0 < device_id_index < len(topic_list):
        return topic_list[device_id_index]
    else:
        return None


def str_is_empty(value):
    '''
    判断参数 value是否为空（空包括：None; ''; 全部为空格）
    :param value:字符串或者None
    '''
    if value == None:
        return True
    if not isinstance(value, str):
        raise ValueError('Input parameter value is not string')
    return value.strip() == ''
