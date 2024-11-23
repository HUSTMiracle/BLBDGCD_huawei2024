import requests
import os
from get_token import get_token
class IotDeviceUtil:
    def __init__(self):
        pass
    def get_device_shadow(self,device_id):
        user_name = os.environ["USER_NAME"]
        user_password = os.environ["USER_PASSWORD"]
        domain_name = os.environ["DOMAIN_NAME"]
        project_name = os.environ["PROJECT_NAME"]
        target_url = os.environ["TOKEN_TARGET_URL"]
        token = get_token(user_name, user_password, domain_name, project_name,target_url)
        url = f'{os.environ["IOT_SERVICE_URL"]}/v5/iot/{os.environ["PROJECT_ID"]}/devices/{device_id}/shadow'
        # Send request.
        headers = {
            'X-Auth-Token': token,
        }
        resp = requests.get(url, headers=headers)
        # Print result.
        #print(resp.status_code)
        #print(resp.text)
        return resp
    def get_all_devices(self):
        user_name = os.environ["USER_NAME"]
        user_password = os.environ["USER_PASSWORD"]
        domain_name = os.environ["DOMAIN_NAME"]
        project_name = os.environ["PROJECT_NAME"]
        target_url = os.environ["TOKEN_TARGET_URL"]
        token = get_token(user_name, user_password, domain_name, project_name,target_url)
        url = f'{os.environ["IOT_SERVICE_URL"]}/v5/iot/{os.environ["PROJECT_ID"]}/devices'
        # Send request.
        headers = {
            'X-Auth-Token': token,
        }
        resp = requests.get(url, headers=headers)

        # Print result.
        print(resp.status_code)
        print(resp.text)
        return resp

if __name__ == '__main__':
    IotDeviceUtil().get_device_shadow("device_id")
    IotDeviceUtil().get_all_devices()