from flask import Blueprint
import controller.iot_controller
prefix = '/api/v1'

iot_controller = controller.iot_controller.IotController()

iot_bp = Blueprint('iot', __name__, url_prefix=prefix)
iot_bp.route('/get_all_devices', methods=['GET'])(iot_controller.ctrl_get_all_devices)
iot_bp.route('/get_device_shadow', methods=['GET'])(iot_controller.ctrl_get_device_shadow)