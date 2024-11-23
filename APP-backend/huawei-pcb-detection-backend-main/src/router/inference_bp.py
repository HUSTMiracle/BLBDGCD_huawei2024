from flask import Blueprint
import controller.inference_controller
prefix = '/api/v1'

inference_controller = controller.inference_controller.InferenceController()

inference_bp = Blueprint('inference', __name__, url_prefix=prefix)
inference_bp.route('/inference', methods=['POST'])(inference_controller.ctrl_inference)
inference_bp.route('/inference_test', methods=['POST'])(inference_controller.ctrl_inference_test)
inference_bp.route('/inference_binary', methods=['POST'])(inference_controller.ctrl_inference_binary)
inference_bp.route('/get_all_histories', methods=['GET'])(inference_controller.ctrl_get_all_histories)
inference_bp.route('/get_one_history', methods=['GET'])(inference_controller.ctrl_get_one_history_result)
