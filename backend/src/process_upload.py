import logging
import os
import glob
import argparse
import time
import requests

import check_channels
import extract_metadata
import create_dzi

import emit
from io_utils import json_io

ROOT_DIR_NAMES = [
    "images",
    "dzi_images",
    "metadata",
    "model"
]

MODEL_DIR_NAMES = [
    "result",
]



def process_upload(image_set_dir):


    try:
        upload_status_path = os.path.join(image_set_dir, "upload_status.json")

        config_path = os.path.join(image_set_dir, "config.json")
        config = json_io.load_json(config_path)

        username = config["username"]
        image_set_name = config["image_set_name"]

        check_channels.check_channels(image_set_dir)


        for root_dir_name in ROOT_DIR_NAMES:
            os.makedirs(os.path.join(image_set_dir, root_dir_name), exist_ok=True)

        for model_dir_name in MODEL_DIR_NAMES:
            os.makedirs(os.path.join(image_set_dir, "model", model_dir_name), exist_ok=True)


        image_names = []
        for image_path in glob.glob(os.path.join(image_set_dir, "images", "*")):
            image_name = os.path.basename(image_path).split(".")[0]
            image_names.append(image_name)


        extract_metadata.extract_metadata(config)

        create_dzi.create_dzi(image_set_dir)

        os.remove(config_path)

        send_prediction_request(username, image_set_name)

        # json_io.save_json(upload_status_path, {"status": "uploaded"})


    except Exception as e:
        
        json_io.save_json(upload_status_path, {"status": "failed", "error": str(e)})

        emit.emit_upload_notification(username, image_set_name)

    # emit.emit_upload_change({
    #     "username": username, 
    #     "image_set_name": image_set_name
    # })

        
def send_prediction_request(username, image_set_name):

    data = {
        "key": username + "/" + image_set_name,
        "task": "predict",
        "request_time": time.time(),
        "username": username,
        "image_set_name": image_set_name
    }

    base_url = "http://" + os.environ.get("FF_IP") + ":" + os.environ.get("FF_PY_PORT") + os.environ.get("FF_PATH")
    req_url = base_url + "/add_request"

    logger = logging.getLogger(__name__)

    logger.info("Emitting {} to {}".format(data, req_url))

    headers = {
        'content-type': 'application/json',
    }

    response = requests.post(req_url, headers=headers, json=data)
    status_code = response.status_code
    json_response = response.json()
    # response.raise_for_status()  # raises exception when not a 2xx response
    if status_code != 200:
        logger.error("Response status code is not 200. Status code: {}".format(status_code))
        logger.error(json_response)
        raise RuntimeError("Bad server response status code.")

    if "message" not in json_response or json_response["message"] != "ok":
        logger.error("Response message is not 'ok'.")
        logger.error(json_response)
        raise RuntimeError("Bad server response.")



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("image_set_dir", type=str)
    
    args = parser.parse_args()
    image_set_dir = args.image_set_dir


    process_upload(image_set_dir)
