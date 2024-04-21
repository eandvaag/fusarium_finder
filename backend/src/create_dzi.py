import logging
import os
import glob
import subprocess
import time
import traceback
import threading
import pyvips

from io_utils import json_io
from lock_queue import LockQueue


accepted_extensions = ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]
NUM_WORKERS = 10

image_queue = LockQueue()
operation_error = False


def dzi_worker(image_set_dir, index):
    global operation_error

    logger = logging.getLogger(__name__)

    images_dir = os.path.join(image_set_dir, "images")
    dzi_images_dir = os.path.join(image_set_dir, "dzi_images")
    image_queue_size = image_queue.size()
    while image_queue_size > 0:
        image_path = image_queue.dequeue()
        image_name = os.path.basename(image_path)
        split_image_name = image_name.split(".")
        dzi_path = os.path.join(dzi_images_dir, split_image_name[0])
        image_extension = split_image_name[-1]

        if (image_extension not in accepted_extensions):
            conv_path = os.path.join(images_dir, split_image_name[0] + ".png")
            try:
                subprocess.run(["convert", image_path, conv_path], check=True)
            except Exception as e:
                trace = traceback.format_exc()
                logger.error("Error from thread {}: {}".format(index, trace))
                operation_error = True
        else:
            conv_path = image_path


        if operation_error:
            logger.info("Thread {}: Error detected, returning.".format(index))
            return

        conv_path = image_path

        try:
            x = pyvips.Image.new_from_file(conv_path)
            x.dzsave(dzi_path)
        except Exception as e:
            trace = traceback.format_exc()
            logger.error("Error from thread {}: {}".format(index, trace))
            operation_error = True
        

        if operation_error:
            logger.info("Thread {}: Error detected, returning.".format(index))
            return

        image_queue_size = image_queue.size()


    return




def create_dzi(image_set_dir):

    logger = logging.getLogger(__name__)

    image_paths = glob.glob(os.path.join(image_set_dir, "images", "*"))

    for image_path in image_paths:
        image_queue.enqueue(image_path)
    
    logger.info("Starting creation of DZI images.")

    start_time = time.time()

    threads = []
    for i in range(NUM_WORKERS):
        x = threading.Thread(target=dzi_worker, args=(image_set_dir, i,))
        threads.append(x)

    for x in threads:
        x.start()

    for x in threads:
        x.join()

    if operation_error:
        raise RuntimeError("Error occurred during DZI image conversion.")
    else:
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info("Finished creation of DZI images. Elapsed time: {}".format(elapsed))