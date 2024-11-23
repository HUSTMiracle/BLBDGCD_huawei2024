import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from flask import request
from util.iot import IotDeviceUtil

class IotController:
    def __init__(self):
        pass
    def ctrl_get_device_shadow(self):
        device_id = request.args.get("device_id")
        resp = IotDeviceUtil().get_device_shadow(device_id)
        if resp.status_code != 200:
            return {
                "success": False,
                "result": resp.json(),
            }
        return {
            "success": True,
            "result": resp.json(),
        }
    def ctrl_get_all_devices(self):
        resp = IotDeviceUtil().get_all_devices()
        if resp.status_code != 200:
            return {
                "success": False,
                "result": resp.json(),
            }
        return {
            "success": True,
            "result": resp.json(),
        }