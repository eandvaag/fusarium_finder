import logging
import argparse
import os
import glob
import shutil
import time
import datetime
import traceback
import threading
import numpy as np
import math as m
import random
from natsort import natsorted
import pandas as pd
import pandas.io.formats.excel
import urllib3

from flask import Flask, request


from io_utils import json_io, tf_record_io

import models.yolov4.driver as yolov4_driver
import emit
from image_wrapper import ImageWrapper
import classifier


cv = threading.Condition()
queue = []
occupied_sets = {}

waiting_workers = 0
TOTAL_WORKERS = 2

app = Flask(__name__)


REQUIRED_JOB_KEYS = [
    "key", 
    "task", 
    "request_time"
]

VALID_TASKS = [
    "predict"
]




def check_job(req):

    for req_key in REQUIRED_JOB_KEYS:
        if not req_key in req:
            raise RuntimeError("Bad request")
        
    if req["task"] not in VALID_TASKS:
        raise RuntimeError("Bad request")
    



@app.route(os.environ.get("FF_PATH") + 'health_request', methods=['POST'])
def health_request():
    return {"message": "alive"}


@app.route(os.environ.get("FF_PATH") + 'is_occupied', methods=['POST'])
def is_occupied():

    logger = logging.getLogger(__name__)
    logger.info("POST to add_request")

    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        job = request.json
        key = job["key"]

        return {"is_occupied": key in occupied_sets}
    else:
        return {"message": 'Content-Type not supported!'}
    

@app.route(os.environ.get("FF_PATH") + 'get_num_workers', methods=['POST'])
def get_num_workers():
    return {"num_workers": str(waiting_workers)}

@app.route(os.environ.get("FF_PATH") + 'add_request', methods=['POST'])
def add_request():
    
    logger = logging.getLogger(__name__)
    logger.info("POST to add_request")

    content_type = request.headers.get('Content-Type')

    if (content_type == 'application/json'):
        job = request.json
        logger.info("Got request: {}".format(job))

        try:
            check_job(job)
        except RuntimeError:
            return {"message": "Job is malformed."}
        
        occupied = False
        with cv:
            if job["key"] in occupied_sets:
                occupied = True
        if occupied:
            return {"message": "The job cannot be enqueued due to an existing job that has not yet been processed."}
        


        with cv:
            occupied_sets[job["key"]] = job
            queue.append(job["key"])
            cv.notify()



        return {"message": "ok"}


    else:
        return {"message": 'Content-Type not supported!'}
        



def job_available():
    return len(queue) > 0


def create_spreadsheet(job):

    username = job["username"]
    image_set_name = job["image_set_name"]

    image_set_dir = os.path.join("usr", "data", username, "image_sets", image_set_name)


    model_dir = os.path.join(image_set_dir, "model")
    result_dir = os.path.join(model_dir, "result")
    prediction_dir = os.path.join(result_dir, "prediction")


    predictions = {}
    image_prediction_paths = glob.glob(os.path.join(prediction_dir, "*"))
    for image_prediction_path in image_prediction_paths:
        image_predictions = json_io.load_json(image_prediction_path)
        image_name = os.path.basename(image_prediction_path)[:-len(".json")]
        predictions[image_name] = {
            "boxes": np.array(image_predictions["boxes"]),
            "detector_scores": np.array(image_predictions["detector_scores"]),
            "classifier_scores": np.array(image_predictions["classifier_scores"]),
            "classes": np.array(image_predictions["classes"])
        }
    
    
    
    columns = [
        "Image Name",
        "Fusarium Percentage",
        "Number of Kernels (Total)",
        "Number of Kernels (Broken)",
        "Number of Kernels (Not Fusarium)",
        "Number of Kernels (Shriveled)",
        "Number of Kernels (Tombstone)",
    ]
    d = {}
    for column in columns:
        d[column] = []
    image_names = natsorted(list(predictions.keys()))

    for image_name in image_names:
        boxes = predictions[image_name]["boxes"]
        scores = predictions[image_name]["detector_scores"]
        classes = predictions[image_name]["classes"]

        score_mask = scores > 0.5

        boxes = boxes[score_mask]
        classes = classes[score_mask]


        num_broken = np.sum(classes == 0)
        num_not_fusarium = np.sum(classes == 1)
        num_shriveled = np.sum(classes == 2)
        num_tombstone = np.sum(classes == 3)

        frac_fus = (num_shriveled + num_tombstone) / (num_shriveled + num_tombstone + num_not_fusarium)
        perc_fus = round(float(frac_fus * 100), 2)


        d["Image Name"].append(image_name)
        d["Fusarium Percentage"].append(perc_fus)
        d["Number of Kernels (Total)"].append(classes.size)
        d["Number of Kernels (Broken)"].append(num_broken)
        d["Number of Kernels (Not Fusarium)"].append(num_not_fusarium)
        d["Number of Kernels (Shriveled)"].append(num_shriveled)
        d["Number of Kernels (Tombstone)"].append(num_tombstone)


    df = pd.DataFrame(data=d, columns=columns)


    out_path = os.path.join(result_dir, "result.csv")
    df.to_csv(out_path, index=False)
    



def process_predict(job):

    logger = logging.getLogger(__name__)

    try:

        username = job["username"]
        image_set_name = job["image_set_name"]

        image_set_dir = os.path.join("usr", "data", username, "image_sets", image_set_name)


        yolov4_driver.predict(job)

        classifier.classify_detections(job)

        create_spreadsheet(job)


        logger.info("Finished predicting for {}".format(job["key"]))

        upload_status_path = os.path.join(image_set_dir, "upload_status.json")
        json_io.save_json(upload_status_path, {"status": "uploaded"})

        with cv:
            del occupied_sets[job["key"]]

        emit.emit_upload_notification(username, image_set_name)


    except Exception as e:
        trace = traceback.format_exc()
        logger.error("Exception occurred in process_predict")
        logger.error(e)
        logger.error(trace)

        if image_set_dir != None:
            upload_status_path = os.path.join(image_set_dir, "upload_status.json")
            if os.path.exists(upload_status_path):
                json_io.save_json(upload_status_path, {"status": "failed", "error": str(e)})

        with cv:
            if job["key"] in occupied_sets:
                del occupied_sets[job["key"]]

        emit.emit_upload_notification(username, image_set_name)




def process_job(job_key):

    logger = logging.getLogger(__name__)
    try:
        job = occupied_sets[job_key]
        task = job["task"]

        if task == "predict":
            process_predict(job)
        else:
            logger.error("Unrecognized task", task)



    except Exception as e:
        trace = traceback.format_exc()
        logger.error("Exception occurred in process_job")
        logger.error(e)
        logger.error(trace)

        with cv:
            if job_key in occupied_sets:
                del occupied_sets[job_key]





def work():
    global waiting_workers

    while True:
        with cv:
            waiting_workers += 1
            cv.wait_for(job_available)
            job_key = queue.pop(0)
            waiting_workers -= 1

        process_job(job_key)







if __name__ == "__main__":

    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #             logical_gpus = tf.config.list_logical_devices('GPU')
    #             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    # exit()

    # # gpus = None


    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    logging.basicConfig(level=logging.INFO)

    for _ in range(TOTAL_WORKERS):
        worker = threading.Thread(target=work)
        worker.start()


    app.run(host=os.environ.get("FF_IP"), port=os.environ.get("FF_PY_PORT"))