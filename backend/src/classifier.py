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
import uuid

from io_utils import json_io
from image_wrapper import ImageWrapper

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

    # model = get_model()

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        # tf.keras.layers.RandomRotation(1.0),
        tf.keras.layers.RandomZoom((-0.15, 0.15), fill_mode="constant"),
        # tf.keras.layers.RandomBrightness(0.2)
    ])

    IMAGE_SIZE = 224
    IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
    # preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    # base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE,
    #                                            include_top=False,
    #                                            weights='imagenet')
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    base_model = tf.keras.applications.MobileNetV3Small(input_shape=IMAGE_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    # preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    # base_model = tf.keras.applications.MobileNetV3Large(input_shape=IMAGE_SHAPE,
    #                                            include_top=False,
    #                                            weights='imagenet')
    # preprocess_input = tf.keras.applications.xception.preprocess_input
    # base_model = tf.keras.applications.Xception(input_shape=IMAGE_SHAPE,
    #                                            include_top=False,
    #                                            weights='imagenet')

    base_model.trainable = True #False

    inputs = tf.keras.Input(shape=IMAGE_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False) #False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(4)(x)
    model = tf.keras.Model(inputs, outputs)






    # model.build(input_shape=(224, 224, 3))
    model.load_weights(weights_path, by_name=False)


    model_dir = os.path.join(image_set_dir, "model")
    result_dir = os.path.join(model_dir, "result")
    prediction_dir = os.path.join(result_dir, "prediction")

    predictions = load_predictions_from_dir(prediction_dir)

    # debug_out_dir = os.path.join(image_set_dir, "debug_classifier_patches")
    # os.makedirs(debug_out_dir)

    # res = np.zeros(shape=(4), dtype=np.int64)
    rev_predictions = {}
    for image_name in predictions.keys():
 
        image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]

        image_wrapper = ImageWrapper(image_path)
        image_array = image_wrapper.load_image_array()

        boxes = predictions[image_name]["boxes"]
        scores = predictions[image_name]["scores"]
        # boxes = boxes[scores > 0.5]

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
                # if crop_height > patch_size or crop_width > patch_size:
                #     # patch_array = tf.image.resize_with_pad(patch_array, patch_size, patch_size)
                #     patch_array = np.array(PILImage.fromarray(crop).crop_pad((patch_size, patch_size)))
                #     # patch_array = tf.cast(patch_array, dtype=tf.float32)\

                if crop_height > patch_size or crop_width > patch_size:

                    if crop_height > crop_width:
                        crop = resize_image(crop, height=patch_size)
                    else:
                        crop = resize_image(crop, width=patch_size)

                    crop_width = crop.shape[1]
                    crop_height = crop.shape[0]


                # else:
                pad_width = (patch_size - crop_width)
                pad_left = pad_width // 2
                pad_height = (patch_size - crop_height)
                pad_bottom = pad_height // 2

                # print("0 crop", crop.dtype, np.max(crop), np.min(crop))

                patch_array = np.zeros(shape=(patch_size, patch_size, 3), dtype=np.uint8)
                patch_array[pad_bottom:pad_bottom+crop.shape[0], pad_left:pad_left+crop.shape[1], :] = crop

                # patch_array = tf.cast(patch_array, dtype=tf.float32)
                # patch_array = tf.image.resize(images=patch_array, size=(224, 224))
                # print("1 patch", patch_array.dtype, np.max(patch_array), np.min(patch_array))


                # out_path = os.path.join(debug_out_dir, str(uuid.uuid4()) + ".png")
                # (PILImage.fromarray(patch_array.astype(np.uint8))).save(out_path)
                patch_array = tf.convert_to_tensor(patch_array, dtype=tf.float32)
                # print("2 tf patch", patch_array.dtype, tf.math.reduce_min(patch_array), tf.math.reduce_max(patch_array))
                
                batch_patch_arrays.append(patch_array)


            batch_patch_arrays = tf.stack(batch_patch_arrays, axis=0)
            # batch_patch_arrays = np.array(batch_patch_arrays)
            
            # print(batch_patch_arrays.shape)
            # print(tf.shape(batch_patch_arrays))
            y_pred = tf.nn.softmax(model.predict_on_batch(batch_patch_arrays))

            # print(tf.shape(y_pred))

            # y_pred = y_pred.numpy()
            # print(y_pred)

            for i in range(tf.shape(y_pred)[0]):
                # cor_cls = tf.argmax(y[i]).numpy()
                pred_cls = tf.argmax(y_pred[i]).numpy()
                cls_score = tf.math.reduce_max(y_pred[i]).numpy()

                classifier_classes.append(int(pred_cls))
                classifier_scores.append(float(cls_score))


                # res[pred_cls] += 1



            # max_inds = np.argmax(y_pred, axis=1)
            # cls_scores = y_pred[np.arange(y_pred.shape[0]), max_inds]

            # classifier_scores.extend(tf.reduce_max(y_pred).numpy().tolist())
            # classifier_classes.extend(tf.argmax(y_pred).numpy().tolist())

            # classifier_scores.extend(cls_scores.tolist())
            # classifier_classes.extend(max_inds.tolist())

            # for i in range(tf.shape(y_pred)[0]):
            #     classifier_class = int(tf.argmax(y_pred[i]).numpy())
            #     classifier_score = float(tf.max(y_pred[i]).numpy())

            #     classifier_scores.append(classifier_score)
            #     classifier_classes.append(classifier_class)
                


        rev_predictions[image_name] = {
            "boxes": predictions[image_name]["boxes"].tolist(),
            "detector_scores": predictions[image_name]["scores"].tolist(),
            "classes": classifier_classes,
            "classifier_scores": classifier_scores
        }

    # print("res", res)

    # validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     directory=debug_out_dir,
    #     labels=None,
    #     label_mode=None,
    #     batch_size=32,
    #     image_size=(224, 224))

    
    # evaluate(model, "val", validation_ds)

    shutil.rmtree(prediction_dir)

    os.makedirs(prediction_dir)
    for image_name in rev_predictions.keys():
        image_predictions_path = os.path.join(prediction_dir, image_name + ".json")
        json_io.save_json(image_predictions_path, rev_predictions[image_name])

    end_time = time.time()
    elapsed = str(datetime.timedelta(seconds=round(end_time - start_time)))
    logger.info("Finished classification. Time elapsed: {}".format(elapsed))
                



def evaluate(model, ds_name, ds):
    res = np.zeros(shape=(4), dtype=np.int64)
    for batch in ds:
        y_pred = tf.nn.softmax(model.predict_on_batch(batch))

        for i in range(tf.shape(y_pred)[0]):
            pred_cls = tf.argmax(y_pred[i]).numpy()
            res[pred_cls] += 1

    print("---")
    print(ds_name)
    print(res)
    print()
