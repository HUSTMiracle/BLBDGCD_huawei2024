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

import threading
import ujson
try:
    import umqtt.robust as mqtt
except ImportError:
    import umqtt as mqtt
from iot_device_sdk_micropython.utils import get_password, get_client_id
from iot_device_sdk_micropython.utils import get_request_id_from_topic, get_device_id_from_topic
from .request import Command, DeviceMessage


class IoTReceivedData:
    def __init__(self, topic, payload):
        self.__topic = topic.decode()
        self.__payload = payload.decode()

    @property
    def topic(self):
        return self.__topic

    @property
    def payload(self):
        return self.__payload

    @property
    def device_id(self):
        return get_device_id_from_topic(self.__topic)

    @property
    def request_id(self):
        return get_request_id_from_topic(self.__topic)

    @property
    def json_payload(self):
        return ujson.loads(self.__payload)


class IotClient(threading.Thread):
    def __init__(self, client_cfg):
        """
        client_cfg：客户信息，包括以下内容：
        server_ip:  IoT平台mqtt对接地址
        device_id:  创建设备时获得的deviceId，
        secret:  创建设备时获得的密钥
        is_ssl: True则建立 ssl 连接
        ssl_certification_path: # ssl证书存放路径
        """
        super(IotClient, self).__init__()
        self.__server_ip = client_cfg.server_ip
        self.__device_id = client_cfg.device_id
        self.__secret = client_cfg.secret
        self.__port = client_cfg.port

        self.__client = mqtt.MQTTClient(
            client_id=get_client_id(self.__device_id),
            server=self.__server_ip,
            port=self.__port,
            user=self.__device_id,
            password=get_password(self.__secret),
            keepalive=60)

        self.__command_callback = None
        self.__device_message_callback = None
        self.__property_set_callback = None
        self.__property_get_callback = None
        self.__user_topic_message_callback = None
        self.__user_defined_topic = list()

    # 启动线程
    def run(self):
        while True:
            self.__client.wait_msg()

    # 建立mqtt连接
    def connect(self):
        print("......Mqtt/Mqtts connecting......")
        # 连接至broker
        self.__client.connect()
        print("-----------------Mqtt/Mqtts connection completed !!!")
        self.__client.set_callback(self.__on_message_received)
        self.__subscribe()

    # 订阅topic
    def __subscribe(self):
        sys_topics = (
            '$oc/devices/{}/sys/commands/#',  # 订阅平台下发命令topic
            '$oc/devices/{}/sys/messages/down',  # 订阅平台消息下发topic
            '$oc/devices/{}/sys/properties/set/#',  # 订阅平台设置设备属性topic
            '$oc/devices/{}/sys/properties/get/#',  # 订阅平台查询设备属性topic
            '$oc/devices/{}/sys/shadow/get/response/#')  # 订阅设备侧主动获取平台设备影子数据的响应topic

        for topic in sys_topics:
            self.__client.subscribe(topic.format(self.__device_id),  qos=1)

    # 订阅topic
    def subscribe(self, topic):
        print("......Subscription topic......")
        if isinstance(topic, str):
            self.__single_subscribe(topic)
        elif isinstance(topic, list):
            self.__batch_subscribe(topic)

    # 批量订阅topic
    def __batch_subscribe(self, topic_list):
        for topic in topic_list:
            self.__single_subscribe(topic)

    # 订阅单个topic
    def __single_subscribe(self, topic):
        print(topic)
        self.__client.subscribe(topic, qos=1)
        print("------You have subscribed: ", topic)

    # 设置回调，根据topic判断，平台操作的类型
    def __on_message_received(self, topic, payload):
        data = IoTReceivedData(topic, payload)

        print()
        print("====== The message is received from the platform ====== ")
        print("Topic: ", data.topic)
        print("npayload: ", data.payload)

        if '/sys/commands/request_id' in topic:
            self.__on_command(data)  # 设备响应平台命令下发
        elif '/sys/messages/down' in topic:
            self.__on_device_message(data)  # 设备响应平台消息下发
        elif '/sys/properties/set/request_id' in topic:
            self.__on_property_set(data)  # 设备响应平台设置设备属性
        elif '/sys/properties/get/request_id' in topic:
            self.__on_property_get(data)  # 设备响应平台查询设备属性
        else:
            self.__on_other(data)

    def report_properties(self, service_properties, qos):
        '''
        设备上报属性:上报json数据，注意serviceId要与Profile中的定义对应
        :param service_properties:服务与属性，参考ServicesProperties类
        :param qos:消息质量等级
        :return:无
        '''
        print("......Device reporting properties......")
        topic = '$oc/devices/' + \
            str(self.__device_id) + '/sys/properties/report'
        payload = {"services": service_properties}
        payload = ujson.dumps(payload)
        self.__client.publish(topic, payload, qos=qos)
        print("-----------------Device report properties completed-----------------")

    # 设备发送消息到平台,又用户自定义topic
    def __publish_raw_message(self, topic, device_message):
        payload = {"content": device_message}
        payload = ujson.dumps(payload)
        self.__client.publish(topic, payload, qos=1)

    # 设备发送消息到平台
    def publish_message(self, *args):
        if len(args) == 1:  # 系统定义topic
            (device_message,) = args
            topic = '$oc/devices/' + \
                str(self.__device_id) + '/sys/messages/up'
            self.__publish_raw_message(topic, device_message)
        elif len(args) == 2:  # 用户自定义topic
            topic, device_message = args
            self.__publish_raw_message(topic, device_message)

    # 响应平台下发的命令
    def respond_command(self, request_id, result_code):
        topic = '$oc/devices/{}/sys/commands/response/request_id={}'.format(
            self.__device_id, request_id)
        payload = {"result_code": result_code}
        payload = ujson.dumps(payload)
        self.__client.publish(topic, payload, qos=1)

    # 响应平台下发的命令
    def respond_device_message(self, data):
        print("------The platform has sent a message------")
        print("------ message topic", data.topic)
        print("------message payload", data.payload)

    # 响应平台设置设备属性
    def respond_property_set(self, request_id, result_code):
        topic = '$oc/devices/{}/sys/properties/set/response/request_id={}'.format(
            self.__device_id, request_id)
        payload = {"result_code": 0, "result_desc": result_code}
        payload = ujson.dumps(payload)
        self.__client.publish(topic, payload, qos=1)

    # 响应平台查询设备属性
    def respond_property_get(self, request_id, service_properties):
        topic = '$oc/devices/{}/sys/properties/get/response/request_id={}'.format(
            self.__device_id, request_id)
        payload = {"services": service_properties}
        payload = ujson.dumps(payload)
        self.__client.publish(topic, payload, qos=1)

    # 平台下发命令后，设备发送响应
    def __on_command(self, data):
        print("-----------------Response command-----------------")
        if self.__command_callback != None and data.device_id in (None, self.__device_id):
            self.__command_callback(
                data.request_id, Command(data.json_payload))
        else:
            self.respond_command(data.request_id, result_code=0)

    # 响应平台下发消息
    def __on_device_message(self, data):
        if self.__device_message_callback != None and data.device_id in (None, self.__device_id):
            self.__device_message_callback(DeviceMessage(data.json_payload))
        else:
            self.respond_device_message(data)

    # 响应设置设备属性
    def __on_property_set(self, data):
        print(
            "-----------------Response platform setting device properties-----------------")
        if self.__property_set_callback != None and data.device_id in (None, self.__device_id):
            self.__property_set_callback(data.request_id, data.payload)
        else:
            self.respond_property_set(data.request_id, result_code="success")

    # 响应设置
    def __on_property_get(self, data):
        print("-----------------Response platform query device properties-----------------")
        if self.__property_get_callback != None and data.device_id in (None, self.__device_id):
            self.__property_get_callback(data.request_id, data.payload)
        else:
            service_id = data.json_payload['service_id']
            self.respond_property_get(
                data.request_id, [{'service_id': service_id}])

    # 处理自定义
    def __on_other(self, data):
        if data.topic in self.__user_defined_topic:
            if self.__user_topic_message_callback != None and data.device_id in (None, self.__device_id):
                self.__user_topic_message_callback(
                    data.request_id, DeviceMessage(data.json_payload))
            else:
                self.respond_device_message(data)
        else:
            print("-----------------This topic is not subscribed-----------------")

    # 用户自定义回调
    def set_command_callback(self, callback):
        self.__command_callback = callback

    def set_device_message_callback(self, callback):
        self.__device_message_callback = callback

    def set_property_set_callback(self, callback):
        self.__property_set_callback = callback

    def set_property_get_callback(self, callback):
        self.__property_get_callback = callback

    def set_user_topic_message_callback(self, callback):
        self.__user_topic_message_callback = callback
