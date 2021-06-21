import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import json

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Processing the data and saving the image and mask """
def process_data(image_path, json_path, save_dir):
    """ Reading the JSON file """
    f = open(json_path, "r")
    data = json.load(f)

    """ Loop over the JSON data (dictionary) """
    for key, value in tqdm(data.items()):
        """ Getting the filename """
        filename = value["filename"]

        """ Extracting the name of the image, by removing its extension """
        name = filename.split(".")[0]

        """ Reading the image """
        image = cv2.imread(f"{image_path}/{filename}", cv2.IMREAD_GRAYSCALE)
        H, W = image.shape

        """ Extracting information about the annotated regions """
        regions = value["regions"]

        if len(regions) == 0:
            mask = np.zeros((H, W))

        else:
            mask = np.zeros((H, W))

            for region in regions:
                cx = int(region["shape_attributes"]["cx"])
                cy = int(region["shape_attributes"]["cy"])
                rx = int(region["shape_attributes"]["rx"])
                ry = int(region["shape_attributes"]["ry"])

                center_coordinates = (cx, cy)
                axes_length = (rx, ry)
                angle = 0
                start_angle = 0
                end_angle = 360
                color = (255, 255, 255)
                thickness = -1

                mask = cv2.ellipse(
                    mask, center_coordinates, axes_length, angle,
                    start_angle, end_angle, color, thickness
                )

        """ Saving the image and mask """
        cv2.imwrite(f"{save_dir}/image/{name}.png", image)
        cv2.imwrite(f"{save_dir}/mask/{name}.png", mask)

if __name__ == "__main__":
    """ Dataset path """
    dataset_path = "dataset"
    dataset = glob(os.path.join(dataset_path, "*"))

    """ Loop over the dataset """
    for data in dataset:
        """ Path for the files """
        image_path = glob(os.path.join(data, "*", "jpg*"))[0]
        json_path = glob(os.path.join(data, "*", "*.json"))[0]

        """ Creating directories to save the data """
        dir_name = data.split("/")[1]
        save_dir = f"data/{dir_name}/"
        create_dir(f"{save_dir}/image")
        create_dir(f"{save_dir}/mask")

        """ Process the data """
        process_data(image_path, json_path, save_dir)
