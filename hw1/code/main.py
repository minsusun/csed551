import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
color_lookup_table = ["blue", "green", "red"]

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
        "visualization_dir": os.path.join(output_dir, "problem2", "visualization"),
        "alpha": 0.3,
        "color_pyplot_config": {
            "row": 2,
            "col": 4,
            "fig_width": 16,
            "fig_height": 8,
        },
        "gray_pyplot_config": {
            "row": 2,
            "col": 2,
            "fig_width": 8,
            "fig_height": 8,
        },
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
    config = configs["problem2"]

    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

    if not os.path.exists(config["visualization_dir"]):
        os.makedirs(config["visualization_dir"])

    for image_filename in config["input_image_list"]:
        is_rgb = "color" in image_filename

        image = cv2.imread(
            os.path.join(config["input_image_base_dir"], image_filename),
            cv2.IMREAD_COLOR if is_rgb else cv2.IMREAD_GRAYSCALE,
        )

        plt_config = config["color_pyplot_config" if is_rgb else "gray_pyplot_config"]

        fig = plt.figure()
        fig.suptitle(f"{image_filename}")
        rows = plt_config["row"]
        cols = plt_config["col"]
        fig.set_figwidth(plt_config["fig_width"])
        fig.set_figheight(plt_config["fig_height"])

        ax = fig.add_subplot(rows, cols, 0 * cols + 1)
        if is_rgb:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(image, cmap="gray")
        ax.set_xlabel("Original Image")
        ax.set_xticks([]), ax.set_yticks([])

        if is_rgb:
            for i in range(3):
                target = image[:, :, i]
                target_flat = target.flatten()

                ax1 = fig.add_subplot(rows, cols, 0 * cols + i + 2)
                ax1.hist(
                    target_flat, 256, color=color_lookup_table[i], alpha=config["alpha"]
                )
                ax1.set_xlabel(f"Original Histogram of {color_lookup_table[i]} channel")
                ax1.get_yaxis().set_visible(False)

                ax2 = ax1.twinx()
                ax2.plot(
                    np.cumsum(np.histogram(target_flat, 256, [0, 256])[0]),
                    color=color_lookup_table[i],
                )
                ax2.get_yaxis().set_visible(False)
        else:
            target_flat = image.flatten()
            ax1 = fig.add_subplot(rows, cols, 0 * cols + 2)
            ax1.hist(target_flat, 256, color="black", alpha=config["alpha"])
            ax1.set_xlabel(f"Original Histogram")
            ax1.get_yaxis().set_visible(False)
            ax2 = ax1.twinx()
            ax2.plot(
                np.cumsum(np.histogram(target_flat, 256, [0, 256])[0]),
                color="black",
            )
            ax2.get_yaxis().set_visible(False)

        processed_image = hw1.histogramEqualization(image)
        cv2.imwrite(
            os.path.join(config["output_dir"], image_filename), processed_image
        )

        ax = fig.add_subplot(rows, cols, 1 * cols + 1)
        if is_rgb:
            ax.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(processed_image, cmap="gray")
        ax.set_xlabel("Histogram Equalized Image")
        ax.set_xticks([]), ax.set_yticks([])

        if is_rgb:
            for i in range(3):
                target = processed_image[:, :, i]
                target_flat = target.flatten()

                ax1 = fig.add_subplot(rows, cols, 1 * cols + i + 2)
                ax1.hist(
                    target_flat, 256, color=color_lookup_table[i], alpha=config["alpha"]
                )
                ax1.set_xlabel(f"Equalized Histogram of {color_lookup_table[i]} channel")
                ax1.get_yaxis().set_visible(False)

                ax2 = ax1.twinx()
                ax2.plot(
                    np.cumsum(np.histogram(target_flat, 256, [0, 256])[0]),
                    color=color_lookup_table[i],
                )
                ax2.get_yaxis().set_visible(False)
        else:
            target_flat = processed_image.flatten()
            ax1 = fig.add_subplot(rows, cols, 1 * cols + 2)
            ax1.hist(target_flat, 256, color="black", alpha=config["alpha"])
            ax1.set_xlabel(f"Equlized Histogram")
            ax1.get_yaxis().set_visible(False)
            ax2 = ax1.twinx()
            ax2.plot(
                np.cumsum(np.histogram(target_flat, 256, [0, 256])[0]),
                color="black",
            )
            ax2.get_yaxis().set_visible(False)

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                config["visualization_dir"],
                f"visualization_{''.join(image_filename.split('.')[:-1])}.png",
            ),
            format="png",
        )


problem1()
problem2()
