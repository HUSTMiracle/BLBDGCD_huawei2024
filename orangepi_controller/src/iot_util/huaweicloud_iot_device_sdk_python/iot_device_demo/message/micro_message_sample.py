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

'''
消息传送demo
包括订阅topic
发布消息
'''

import logging
import utime
from iot_device_sdk_micropython.client import IoTClientConfig
from iot_device_sdk_micropython.client import IotClient

# 日志设置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run():
    # 客户端配置
    client_cfg = IoTClientConfig(
        # 替换为真正的接入地址、device id和密钥，参考readme_CN.md中“上传产品模型并注册设备”
        server_ip='access address',
        device_id='your device id',
        secret='your secret')
    # 创建设备
    iot_client = IotClient(client_cfg)
    iot_client.connect()  # 建立连接

    # 设备接受平台下发消息的响应
    def message_callback(device_message):
        logger.info(('device message, device id:  ', device_message.device_id))
        logger.info(('device message, id = ', device_message.id))
        logger.info(('device message, name: ', device_message.name))
        logger.info(('device message. content: ', device_message.content))

    # 设置平台下发消息响应的回调
    iot_client.set_device_message_callback(message_callback)
    # 设置平台下发自定义topic消息响应的回调
    iot_client.set_user_topic_message_callback(message_callback)

    '''
    订阅自定义topic, 需提前在平台配置自定义topic
    支持批量订阅（topic存放列表中），和逐个订阅（单个topic,无需放入列表）
    '''
    topics = ['$oc/devices/{}/user/user_message/up',
              '$oc/devices/{}/user/myself_prop/up',
              '$oc/devices/{}/user/wpy/up']
    topics = [t.format(client_cfg.device_id) for t in topics]
    iot_client.subscribe(topic=topics)

    # 发送自定义topic消息
    iot_client.publish_message(topics[0], 'Hello Huawei cloud IoT')

    # 设备向平台发送消息，系统默认topic
    iot_client.publish_message('raw message: Hello Huawei cloud IoT')

    iot_client.start()  # 线程启动

    while True:
        tem = utime.time() % 25
        hum = utime.time() % 35
        lum = utime.time() % 100
        services = [{
            "service_id": "Agriculture",
            "properties": {
                "Temperature": tem,
                "Humidity": hum,
                "Luminance": lum,
                "LightStatus": "ON",
                "MotorStatus": "OFF"
            }
        }]

        iot_client.report_properties(services, 0)
        utime.sleep_ms(2000)


if __name__ == '__main__':
    run()
