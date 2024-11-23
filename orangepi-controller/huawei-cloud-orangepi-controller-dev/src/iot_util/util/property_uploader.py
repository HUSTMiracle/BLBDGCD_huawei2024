import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../huaweicloud_iot_device_sdk_python'))
import json
import logging
import time
from huaweicloud_iot_device_sdk_python.iot_device_sdk_python.client.client_conf import ClientConf
from huaweicloud_iot_device_sdk_python.iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from huaweicloud_iot_device_sdk_python.iot_device_sdk_python.iot_device import IotDevice
from huaweicloud_iot_device_sdk_python.iot_device_sdk_python.client.listener.property_listener import PropertyListener
from huaweicloud_iot_device_sdk_python.iot_device_sdk_python.client.listener.default_publish_action_listener import DefaultPublishActionListener
from huaweicloud_iot_device_sdk_python.iot_device_sdk_python.client.request.service_property import ServiceProperty
from huaweicloud_iot_device_sdk_python.iot_device_sdk_python.client import iot_result
from huaweicloud_iot_device_sdk_python.iot_device_sdk_python.transport.connect_listener import ConnectListener
logging.basicConfig(level=logging.WARN,
                    format="%(asctime)s - %(threadName)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class CustomConnectListener(ConnectListener):

    def __init__(self, iot_device: IotDevice):
        """ 传入一个IotDevice实例 """
        self.device = iot_device

    def connection_lost(self, cause: str):
        """
        连接丢失通知

        Args:
            cause:   连接丢失原因
        """
        logger.warning("connection lost. cause: " + cause)
        logger.warning("you can define reconnect in this method.")

    def connect_complete(self, reconnect: bool, server_uri: str):
        """
        连接成功通知，如果是断链重连的情景，重连成功会上报断链的时间戳

        Args:
            reconnect:   是否为重连（当前此参数没有作用）
            server_uri:  服务端地址
        """
        logger.info("connect success. server uri is " + server_uri)



class PropertyUploader:
    def __init__(self):
        server_uri = os.environ["SERVER_URI"]   # 需要改为用户保存的接入地址
        port = 8883
        device_id = os.environ["DEVICE_ID"]
        secret = os.environ["SECRET"]
        # iot平台的CA证书，用于服务端校验
        iot_ca_cert_path = os.environ["IOT_CA_CERT_PATH"]
        
        connect_auth_info = ConnectAuthInfo()
        connect_auth_info.server_uri = server_uri
        connect_auth_info.port = port
        connect_auth_info.id = device_id
        connect_auth_info.secret = secret
        connect_auth_info.iot_cert_path = iot_ca_cert_path
        connect_auth_info.check_timestamp = "0"
        connect_auth_info.bs_mode = ConnectAuthInfo.BS_MODE_DIRECT_CONNECT
        connect_auth_info.reconnect_on_failure = True #重连设置
        connect_auth_info.min_backoff = 1 * 1000
        connect_auth_info.max_backoff = 30 * 1000
        connect_auth_info.max_buffer_message = 100
        self.client_conf = ClientConf(connect_auth_info)

        self.device = IotDevice(self.client_conf)
        if self.device.connect() != 0:
            logger.error("init failed")
        logger.info("init successful")
        self.device.get_client().add_connect_listener(CustomConnectListener(self.device))
    def upload_properties(self,service_id:str,properties: dict):
        service_property = ServiceProperty()
        #service_properties_message = DeviceMessage()
        service_property.service_id = service_id
        service_property.properties = properties
        services = [service_property]
        self.device.get_client().report_properties(services, DefaultPublishActionListener())
