import os
import requests
import logging


RUNNING_IN_APPTAINER = "RUNNING_IN_APPTAINER" in os.environ and os.environ["RUNNING_IN_APPTAINER"] == "yes"
if RUNNING_IN_APPTAINER:
    base_url = ""
else:
    base_url = "https://" + os.environ.get("FF_IP") + ":" + os.environ.get("FF_PORT") + os.environ.get("FF_PATH")
upload_notification_url = base_url + "/upload_notification"



def emit_upload_notification(username, image_set_name):
    emit(upload_notification_url, {"username": username, "image_set_name": image_set_name})

def emit(url, data):
    logger = logging.getLogger(__name__)

    if RUNNING_IN_APPTAINER:
        return True

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
