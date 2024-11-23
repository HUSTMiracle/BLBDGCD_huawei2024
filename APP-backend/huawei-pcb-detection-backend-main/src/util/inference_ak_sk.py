# coding=utf-8
# 只能处理12M以下的数据
import requests
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from apig_sdk import signer

class InferenceAKSK:
    def __init__(self):
        pass
    def infer(self,file_path):
        # Config url, ak, sk and file path.
        url = os.environ["ONLINE_SERVICE_URL"]
        # 认证用的ak和sk硬编码到代码中或者明文存储都有很大的安全风险,建议在配置文件或者环境变量中密文存放,使用时解密,确保安全;
        # 本示例以ak和sk保存在环境变量中来实现身份验证为例,运行本示例前请先在本地环境中设置环境变量HUAWEICLOUD_SDK_AK和HUAWEICLOUD_SDK_SK。
        ak = os.environ["HUAWEICLOUD_SDK_AK"]
        sk = os.environ["HUAWEICLOUD_SDK_SK"]

        # Create request, set method, url, headers and body.
        method = 'POST'
        headers = {"x-sdk-content-sha256": "UNSIGNED-PAYLOAD"}
        request = signer.HttpRequest(method, url, headers)

        # Create sign, set the AK/SK to sign and authenticate the request.
        sig = signer.Signer()
        sig.Key = ak
        sig.Secret = sk
        sig.Sign(request)

        # Send request
        files = {'images': open(file_path, 'rb')}
        resp = requests.request(request.method, request.scheme + "://" + request.host + request.uri, headers=request.headers, files=files)
        print(resp.json())
        # return result
        return resp.json()
