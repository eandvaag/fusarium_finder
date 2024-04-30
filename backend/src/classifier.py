import logging
import os
import glob
import numpy as np
import cv2
import tensorflow as tf
import time
import datetime
from PIL import Image as PILImage
import shutil


from io_utils import json_io
from image_wrapper import ImageWrapper
import emit

def load_predictions_from_dir(predictions_dir):
    predictions = {}
    image_prediction_paths = glob.glob(os.path.join(predictions_dir, "*"))
    for image_prediction_path in image_prediction_paths:
        image_predictions = json_io.load_json(image_prediction_path)
        image_name = os.path.basename(image_prediction_path)[:-len(".json")]
        predictions[image_name] = {
            "boxes": np.array(image_predictions["boxes"]),
            "scores": np.array(image_predictions["scores"]),
            "classes": np.array(image_predictions["classes"])
        }
    return predictions


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized





def classify_detections(job):

    logger = logging.getLogger(__name__)

    start_time = time.time()

    username = job["username"]
    image_set_name = job["image_set_name"]

    tf.keras.backend.clear_session()

    image_set_dir = os.path.join("usr", "data", username, "image_sets", image_set_name)

    classifier_dir = os.path.join("usr", "shared", "classifier")
    weights_dir = os.path.join(classifier_dir, "weights")

    weights_path = os.path.join(weights_dir, "weights.h5")

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomZoom((-0.15, 0.15), fill_mode="constant"),
    ])

    IMAGE_SIZE = 224
    IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    base_model = tf.keras.applications.MobileNetV3Small(input_shape=IMAGE_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

    base_model.trainable = True

    inputs = tf.keras.Input(shape=IMAGE_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(4)(x)
    model = tf.keras.Model(inputs, outputs)

    model.load_weights(weights_path, by_name=False)


    model_dir = os.path.join(image_set_dir, "model")
    result_dir = os.path.join(model_dir, "result")
    prediction_dir = os.path.join(result_dir, "prediction")

    predictions = load_predictions_from_dir(prediction_dir)

    total_num_boxes = 0
    for image_name in predictions.keys():
        total_num_boxes += len(predictions[image_name]["boxes"])


    num_processed = 0
    rev_predictions = {}
    prev_percent_complete = 0
    for image_name in predictions.keys():
 
        image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]

        image_wrapper = ImageWrapper(image_path)
        image_array = image_wrapper.load_image_array()

        boxes = predictions[image_name]["boxes"]

        patch_size = 224
        classifier_scores = []
        classifier_classes = []
        batch_size = 64



        for start_ind in range(0, boxes.shape[0], batch_size):
            batch_patch_arrays = []
            for box_ind in range(start_ind, min(start_ind + batch_size, boxes.shape[0]), 1):
                box = boxes[box_ind]

                crop = image_array[box[0]:box[2], box[1]:box[3], :]
                crop_width = crop.shape[1]
                crop_height = crop.shape[0]

                if crop_height > patch_size or crop_width > patch_size:

                    if crop_height > crop_width:
                        crop = resize_image(crop, height=patch_size)
                    else:
                        crop = resize_image(crop, width=patch_size)

                    crop_width = crop.shape[1]
                    crop_height = crop.shape[0]


                pad_width = (patch_size - crop_width)
                pad_left = pad_width // 2
                pad_height = (patch_size - crop_height)
                pad_bottom = pad_height // 2

                patch_array = np.zeros(shape=(patch_size, patch_size, 3), dtype=np.uint8)
                patch_array[pad_bottom:pad_bottom+crop.shape[0], pad_left:pad_left+crop.shape[1], :] = crop

                patch_array = tf.convert_to_tensor(patch_array, dtype=tf.float32)

                batch_patch_arrays.append(patch_array)


            batch_patch_arrays = tf.stack(batch_patch_arrays, axis=0)

            y_pred = tf.nn.softmax(model.predict_on_batch(batch_patch_arrays))

            for i in range(tf.shape(y_pred)[0]):
                pred_cls = tf.argmax(y_pred[i]).numpy()
                cls_score = tf.math.reduce_max(y_pred[i]).numpy()

                classifier_classes.append(int(pred_cls))
                classifier_scores.append(float(cls_score))

            num_processed += int(tf.shape(y_pred)[0])
            percent_complete = round((num_processed / total_num_boxes) * 100)
            if percent_complete > prev_percent_complete:
                emit.emit_image_set_progress_update(username, 
                                                    image_set_name, 
                                                    "Running Classifier (" + str(percent_complete) + "% Complete)") 
                prev_percent_complete = percent_complete

        rev_predictions[image_name] = {
            "boxes": predictions[image_name]["boxes"].tolist(),
            "detector_scores": predictions[image_name]["scores"].tolist(),
            "classes": classifier_classes,
            "classifier_scores": classifier_scores
        }

    shutil.rmtree(prediction_dir)

    os.makedirs(prediction_dir)
    for image_name in rev_predictions.keys():
        image_predictions_path = os.path.join(prediction_dir, image_name + ".json")
        json_io.save_json(image_predictions_path, rev_predictions[image_name])

    end_time = time.time()
    elapsed = str(datetime.timedelta(seconds=round(end_time - start_time)))
    logger.info("Finished classification. Time elapsed: {}".format(elapsed))