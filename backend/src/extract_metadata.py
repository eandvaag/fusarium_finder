import os
import glob
import tqdm
import natsort

from image_wrapper import ImageWrapper
from io_utils import json_io



MULTIPLE_CAMERA_TYPES_MESSAGE = "The images in the image set were captured by several different camera types. This is not allowed."



def extract_metadata(config):

    username = config["username"]
    image_set_name = config["image_set_name"]


    image_set_dir = os.path.join("usr", "data", username, "image_sets", 
                                 image_set_name)



    images_dir = os.path.join(image_set_dir, "images")
    metadata_dir = os.path.join(image_set_dir, "metadata")

    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    metadata_path = os.path.join(metadata_dir, "metadata.json")

    if os.path.exists(metadata_path):
        raise RuntimeError("Existing metadata file found.")


    image_set_metadata = {
        "images": {},
    }


    image_num = 0
    for image_path in tqdm.tqdm(glob.glob(os.path.join(images_dir, "*")), desc="Extracting metadata"):

        image_name = os.path.basename(image_path).split(".")[0]

        image = ImageWrapper(image_path)

        image_width, image_height = image.get_wh()


        image_set_metadata["images"][image_name] = {
            "width_px": image_width,
            "height_px": image_height
        }


    json_io.save_json(metadata_path, image_set_metadata)

