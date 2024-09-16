import os
import cv2
import numpy as np

import hw1
import util

input_dir = os.path.join("..", "images/input")
output_dir = os.path.join("..", "images/output")

color_image_list = [
    filename for filename in os.listdir(input_dir) if "color" in filename
]
gray_image_list = [filename for filename in os.listdir(input_dir) if "gray" in filename]
border_type_lookup_table = {
    cv2.BORDER_CONSTANT: "constant",
    cv2.BORDER_REPLICATE: "replicate",
    cv2.BORDER_REFLECT: "reflect",
    cv2.BORDER_WRAP: "wrap",
    cv2.BORDER_REFLECT_101: "reflect-101",
}

configs = {
    "problem1": {
        "input_image_base_dir": input_dir,
        "input_image_list": color_image_list,
        "border_type_list": [
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
        ],
        "kernel_size_list": [1, 25, 45],
        "kernel_sigma_list": [0.1, 1, 10],
        "separable_list": [True, False],
        "output_dir": os.path.join(output_dir, "problem1"),
    },
    "problem2": {
        "input_image_base_dir": input_dir,
        "input_image_list": color_image_list + gray_image_list,
        "output_dir": os.path.join(output_dir, "problem2"),
    },
}


def problem1():
    @util.profile
    def work(**kwargs):
        return hw1.filterGaussian(**kwargs)

    config = configs["problem1"]

    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

    for image in config["input_image_list"]:
        for kernel_size in config["kernel_size_list"]:
            for kernel_sigma in config["kernel_sigma_list"]:
                for border_type in config["border_type_list"]:
                    for separable in config["separable_list"]:
                        original_image = cv2.imread(
                            os.path.join(config["input_image_base_dir"], image)
                        )
                        print(f"[ ! ]Done reading {image}!")
                        processed_image = work(
                            image=original_image,
                            kernel_size=kernel_size,
                            kernel_sigma=kernel_sigma,
                            border_type=border_type,
                            separable=separable,
                        )
                        print(f"[ ! ]Applied Gaussian filter on image!")
                        output_file_name = f"{''.join(image.split('.')[:-1])}_{kernel_size}_{kernel_sigma}_{border_type_lookup_table[border_type]}_{separable}.{image.split('.')[-1]}"
                        cv2.imwrite(
                            os.path.join(config["output_dir"], output_file_name),
                            processed_image,
                        )
                        print(f"[***]Saved processed image as {output_file_name}!")
                        print(f"[***]  kernel_size: {kernel_size}")
                        print(f"[***]  kernel_sigma: {kernel_sigma}")
                        print(f"[***]  border_type: {border_type_lookup_table[border_type]}")
                        print(f"[***]  border_type: {separable}")
                        print()

def problem2():
    config = configs["problem1"]

    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

problem1()
problem2()
