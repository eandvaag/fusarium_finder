import os
import requests
import logging

from io_utils import json_io


base_url = "https://" + os.environ.get("FF_IP") + ":" + os.environ.get("FF_PORT") + os.environ.get("FF_PATH")
upload_notification_url = base_url + "upload_notification"
progress_notification_url = base_url + "progress_notification"



def emit_upload_notification(username, image_set_name):
    emit(upload_notification_url, {"username": username, "image_set_name": image_set_name})

def emit_image_set_progress_update(username, image_set_name, progress_message):
    image_set_dir = os.path.join("usr", "data", username, "image_sets", image_set_name)
    upload_status_path = os.path.join(image_set_dir, "upload_status.json")
    json_io.save_json(upload_status_path, {"status": "processing", "progress": progress_message})
    emit(progress_notification_url, {"username": username, "image_set_name": image_set_name, "progress": progress_message})


def emit(url, data):
    logger = logging.getLogger(__name__)

    logger.info("Emitting {} to {}".format(data, url))
    headers = {'API-Key': os.environ["FF_API_KEY"]}

    response = requests.post(url, data=data, headers=headers, verify=False)
    status_code = response.status_code
    json_response = response.json()
    # response.raise_for_status()  # raises exception when not a 2xx response
    if status_code != 200:
        logger.error("Response status code is not 200. Status code: {}".format(status_code))
        logger.error(json_response)
        return False

    if "message" not in json_response or json_response["message"] != "received":
        logger.error("Response message is not 'received'.")
        logger.error(json_response)
        return False


    return True
