import requests
import os
from get_token import get_token
class InferenceToken:
    def __init__(self):
        pass
    def infer(self,file_path):
        user_name = os.environ["USER_NAME"]
        user_password = os.environ["USER_PASSWORD"]
        domain_name = os.environ["DOMAIN_NAME"]
        project_name = os.environ["PROJECT_NAME"]
        target_url = os.environ["TOKEN_TARGET_URL"]
        token = get_token(user_name, user_password, domain_name, project_name,target_url)
        url = os.environ["ONLINE_SERVICE_URL"]
        # Send request.
        headers = {
            'X-Auth-Token': token
        }
        files = {
            'images': open(file_path, 'rb')
        }
        resp = requests.post(url, headers=headers, files=files)

        # Print result.
        print(resp.status_code)
        print(resp.text)
        return resp.json()