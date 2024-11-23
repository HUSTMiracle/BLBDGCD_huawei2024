import requests
def get_token(user_name, user_password, domain_name, project_name,target_url):
        data = {
            "auth": {
                "identity": {
                    "methods": [
                        "password"
                    ],
                    "password": {
                        "user": {
                            "name": user_name,
                            "password": user_password,
                            "domain": {
                                "name": domain_name
                            }
                        }
                    }
                },
                "scope": {
                    "project": {
                        "name": project_name
                    }
                }
            }
        }
        response = requests.post(target_url, json=data)
        return response.headers.get("X-Subject-Token")