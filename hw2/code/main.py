import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

import util
import hw2

input_dir = os.path.join("..", "images", "input")
output_dir = os.path.join("..", "images", "output")

image_list = ["color1.jpg", "color3.jpg", "color4.jpg"]
lowpass_filter_padding_list = [10, 20, 30]
lowpass_filter_threshold_list = [10, 25, 50]
color_lookup_table = ["Blue", "Green", "Red"]

configs = {
    "problem1": {
        "input_image_base_dir": input_dir,
        "input_image_list": image_list,
        "output_dir": os.path.join(output_dir, "problem1"),
        "supplementary_output_dir": os.path.join(
            output_dir, "problem1", "supplementary"
        ),
        "padding_list": lowpass_filter_padding_list,
        "threshold_list": lowpass_filter_threshold_list,
    },
    "problem2": {  # match configs with problem 1 to compare
        "input_image_base_dir": input_dir,
        "input_image_list": image_list,
        "output_dir": os.path.join(output_dir, "problem2"),
        "supplementary_output_dir": os.path.join(
            output_dir, "problem2", "supplementary"
        ),
        "padding_list": lowpass_filter_padding_list,
        "threshold_list": lowpass_filter_threshold_list,
    },
    "problem3": {
        "input_image_base_dir": input_dir,
        "input_image_list": image_list,
        "output_dir": os.path.join(output_dir, "problem3"),
        "supplementary_output_dir": os.path.join(
            output_dir, "problem3", "supplementary"
        ),
        "alpha_list": [1.0, 2.0, 3.0],
        "padding_list": [10, 25, 50],
        "domain_list": ["spatial", "frequency"],
    },
}


def problem1():
    config = configs["problem1"]

    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])
    if not os.path.exists(config["supplementary_output_dir"]):
        os.makedirs(config["supplementary_output_dir"])

    for image_filename in config["input_image_list"]:
        original_image = cv2.imread(
            os.path.join(config["input_image_base_dir"], image_filename)
        )

        print(f"[ ! ]Done reading {image_filename}!")
        print()

        # input image visualization in frequency domain
        fig = plt.figure()
        fig.set_figwidth(20)
        fig.set_figheight(6)
        fig.suptitle(f"Original {image_filename} in Frequency Domain", fontsize=25)

        for i in range(3):
            ax = fig.add_subplot(1, 3, i + 1)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_xlabel(f"{color_lookup_table[i]} Channel", fontsize=15)
            data = np.log(np.abs(np.fft.fftshift(np.fft.fft2(original_image[:, :, i]))))
            ax.imshow(data, cmap="gray")

        original_image_fft_filename = "".join(image_filename.split(".")[:-1]) + ".png"
        original_image_fft_file_path = os.path.join(
            config["supplementary_output_dir"], original_image_fft_filename
        )
        fig.tight_layout()
        fig.savefig(original_image_fft_file_path, format="png")

        print(
            f'[ ! ]Saved visualization of original image in frequency domain at "{original_image_fft_file_path}"'
        )
        print()

        # experiments
        for padding in config["padding_list"]:
            for threshold in config["threshold_list"]:
                print(f"[***]Target {image_filename}")
                print(f"[***]    Performing {hw2.idealLowPassFiltering.__name__}")
                print(f"[***]    padding size: {padding}")
                print(f"[***]    threshold: {threshold:.2f}")

                filtered_image = hw2.idealLowPassFiltering(
                    original_image, padding, threshold, False
                )
                print(f"[ ! ]Applied Ideal low pass filter on image!")

                output_filename = f"{''.join(image_filename.split('.')[:-1])}_{padding}_{threshold}.png"
                output_file_path = os.path.join(config["output_dir"], output_filename)

                cv2.imwrite(
                    output_file_path,
                    filtered_image[padding:-padding, padding:-padding, :],
                )
                print(f'[ ! ]Saved filtered image at "{output_file_path}"')

                # filtered image visualization
                fig = plt.figure()
                fig.set_figwidth(20)
                fig.set_figheight(6)
                fig.suptitle(
                    f"Filtered {image_filename} in Frequency Domain", fontsize=25
                )

                for i in range(3):
                    ax = fig.add_subplot(1, 3, i + 1)
                    ax.set_xticks([]), ax.set_yticks([])
                    ax.set_xlabel(f"{color_lookup_table[i]} Channel", fontsize=15)
                    data = np.log(
                        np.abs(np.fft.fftshift(np.fft.fft2(filtered_image[:, :, i])))
                    )
                    ax.imshow(data, cmap="gray")

                filtered_image_fft_filename = f"{''.join(image_filename.split('.')[:-1])}_{padding}_{threshold}.png"
                filtered_image_fft_file_path = os.path.join(
                    config["supplementary_output_dir"], filtered_image_fft_filename
                )
                fig.savefig(filtered_image_fft_file_path, format="png")

                print(
                    f'[ ! ]Saved visualization of filtered image in frequency domain at "{filtered_image_fft_file_path}"'
                )
                print()

    for threshold in config["threshold_list"]:
        # filter visualization
        filter = (
            hw2.idealLowPassFilter((200, 200), threshold) * 255
        )  # normalize
        filter_filename = f"{threshold}.png"
        filter_file_path = os.path.join(
            config["supplementary_output_dir"], filter_filename
        )
        cv2.imwrite(filter_file_path, filter)
        print(
            f'[ ! ]Saved visualization of filter in frequency domain at "{filter_file_path}"'
        )
    print()


def problem2():
    config = configs["problem2"]

    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])
    if not os.path.exists(config["supplementary_output_dir"]):
        os.makedirs(config["supplementary_output_dir"])

    for image_filename in config["input_image_list"]:
        original_image = cv2.imread(
            os.path.join(config["input_image_base_dir"], image_filename)
        )

        print(f"[ ! ]Done reading {image_filename}!")
        print()

        # input image visualization in frequency domain
        fig = plt.figure()
        fig.set_figwidth(20)
        fig.set_figheight(6)
        fig.suptitle(f"Original {image_filename} in Frequency Domain", fontsize=25)

        for i in range(3):
            ax = fig.add_subplot(1, 3, i + 1)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_xlabel(f"{color_lookup_table[i]} Channel", fontsize=15)
            data = np.log(np.abs(np.fft.fftshift(np.fft.fft2(original_image[:, :, i]))))
            ax.imshow(data, cmap="gray")

        original_image_fft_filename = "".join(image_filename.split(".")[:-1]) + ".png"
        original_image_fft_file_path = os.path.join(
            config["supplementary_output_dir"], original_image_fft_filename
        )
        fig.savefig(original_image_fft_file_path, format="png")

        print(
            f'[ ! ]Saved visualization of original image in frequency domain at "{original_image_fft_file_path}"'
        )
        print()

        # experiments
        for padding in config["padding_list"]:
            for threshold in config["threshold_list"]:
                print(f"[***]Target {image_filename}")
                print(f"[***]    Performing {hw2.gaussianLowPassFiltering.__name__}")
                print(f"[***]    padding size: {padding}")
                print(f"[***]    threshold: {threshold:.2f}")

                filtered_image = hw2.gaussianLowPassFiltering(
                    original_image, padding, threshold, False
                )
                print(f"[ ! ]Applied Gaussian low pass filter on image!")

                output_filename = f"{''.join(image_filename.split('.')[:-1])}_{padding}_{threshold}.png"
                output_file_path = os.path.join(config["output_dir"], output_filename)

                cv2.imwrite(
                    output_file_path,
                    filtered_image[padding:-padding, padding:-padding, :],
                )
                print(f'[ ! ]Saved filtered image at "{output_file_path}"')

                # filtered image visualization
                fig = plt.figure()
                fig.set_figwidth(20)
                fig.set_figheight(6)
                fig.suptitle(
                    f"Filtered {image_filename} in Frequency Domain", fontsize=25
                )

                for i in range(3):
                    ax = fig.add_subplot(1, 3, i + 1)
                    ax.set_xticks([]), ax.set_yticks([])
                    ax.set_xlabel(f"{color_lookup_table[i]} Channel", fontsize=15)
                    data = np.log(
                        np.abs(np.fft.fftshift(np.fft.fft2(filtered_image[:, :, i])))
                    )
                    ax.imshow(data, cmap="gray")

                filtered_image_fft_filename = f"{''.join(image_filename.split('.')[:-1])}_{padding}_{threshold}.png"
                filtered_image_fft_file_path = os.path.join(
                    config["supplementary_output_dir"], filtered_image_fft_filename
                )
                fig.savefig(filtered_image_fft_file_path, format="png")

                print(
                    f'[ ! ]Saved visualization of filtered image in frequency domain at "{filtered_image_fft_file_path}"'
                )
                print()

    for threshold in config["threshold_list"]:
        # filter visualization
        filter = (
            hw2.gaussianLowPassFilter((200, 200), threshold) * 255
        )  # normalize
        filter_filename = f"{threshold}.png"
        filter_file_path = os.path.join(
            config["supplementary_output_dir"], filter_filename
        )
        cv2.imwrite(filter_file_path, filter)
        print(
            f'[ ! ]Saved visualization of filter in frequency domain at "{filter_file_path}"'
        )
    print()                


def problem3():
    @util.profile
    def work(**kwargs):
        return hw2.unsharpMasking(**kwargs)

    config = configs["problem3"]

    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])
    if not os.path.exists(os.path.join(config["output_dir"], "spatial")):
        os.makedirs(os.path.join(config["output_dir"], "spatial"))
    if not os.path.exists(os.path.join(config["output_dir"], "frequency")):
        os.makedirs(os.path.join(config["output_dir"], "frequency"))

    for image_filename in config["input_image_list"]:
        original_image = cv2.imread(
            os.path.join(config["input_image_base_dir"], image_filename)
        )

        print(f"[ ! ]Done reading {image_filename}!")
        print()

        for alpha in config["alpha_list"]:
            for padding in config["padding_list"]:
                sigma = round(padding / 6.0, 1) # follow material configuration

                for domain in config["domain_list"]:
                    # main process
                    print(f"[***]Target {image_filename}")
                    print(f"[***]    Performing {hw2.unsharpMasking.__name__}")
                    print(f"[***]    alpha: {alpha}")
                    print(f"[***]    padding size: {padding}")
                    print(f"[***]    sigma: {sigma:.2f}")
                    print(f"[***]    doamin: {domain}")

                    result_image = work(
                        image=original_image,
                        alpha=alpha,
                        padding=padding,
                        sigma=sigma,
                        domain=domain,
                    )
                    print(f"[ ! ]Applied Unsharpening on image")

                    result_image_filename = f"{''.join(image_filename.split('.')[:-1])}_{alpha}_{padding}_{sigma}.png"
                    result_image_file_path = os.path.join(
                        config["output_dir"], domain, result_image_filename
                    )
                    cv2.imwrite(
                        result_image_file_path, result_image
                    )  # ~/problem3/domain/filename.png
                    print(f"[ ! ]Saved unsharpened image at \"{result_image_file_path}\"")

                    # visualization
                    height, width, channel = original_image.shape
                    filter_size = 2 * padding + 1

                    fig = plt.figure()

                    filter = hw2.gauss2d((filter_size, filter_size), sigma)  # visualize

                    padded_image = cv2.copyMakeBorder(
                        original_image,
                        padding,
                        padding,
                        padding,
                        padding,
                        cv2.BORDER_REPLICATE,
                    )  # visualize

                    if domain == "spatial":
                        fig.suptitle("Spatial Domain Unsharpening")
                        fig.set_figwidth(6)
                        fig.set_figheight(2)

                        ax = fig.add_subplot(1, 4, 1)
                        ax.set_xticks([])
                        ax.get_yaxis().set_visible(False)
                        ax.set_xlabel("Filter")
                        ax.imshow(filter, cmap="gray")

                        ax = fig.add_subplot(1, 4, 2)
                        ax.set_xticks([])
                        ax.get_yaxis().set_visible(False)
                        ax.set_xlabel("Padded")
                        ax.imshow(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))

                        filtered_image = np.zeros_like(original_image).astype(
                            np.float32
                        )  # visualize

                        for i in range(height):
                            for j in range(width):
                                for k in range(channel):
                                    filtered_image[i, j, k] = np.sum(
                                        padded_image[
                                            i : i + filter_size :,
                                            j : j + filter_size,
                                            k,
                                        ]
                                        * filter
                                    )

                        ax = fig.add_subplot(1, 4, 3)
                        ax.set_xticks([])
                        ax.get_yaxis().set_visible(False)
                        ax.set_xlabel("Filtered")
                        ax.imshow(
                            cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB).astype(
                                np.uint8
                            )
                        )

                        high_frequency_image = (
                            original_image - filtered_image
                        )  # visualize
                        high_frequency_image = np.clip(high_frequency_image, 0, 255)
                        ax = fig.add_subplot(1, 4, 4)
                        ax.set_xticks([])
                        ax.get_yaxis().set_visible(False)
                        ax.set_xlabel(f"High freq")
                        ax.imshow(
                            cv2.cvtColor(
                                high_frequency_image, cv2.COLOR_BGR2RGB
                            ).astype(np.uint8)
                        )
                    else:  # frequency domain
                        fig.suptitle("Frequency Domain Unsharpening")
                        fig.set_figheight(5)
                        fig.set_figwidth(10)

                        ax = fig.add_subplot(3, 7, 8)
                        ax.set_xticks([])
                        ax.set_xlabel("Filter")
                        ax.get_yaxis().set_visible(False)
                        ax.imshow(filter, cmap="gray")

                        ax = fig.add_subplot(3, 7, 9)
                        ax.set_xticks([])
                        ax.get_yaxis().set_visible(False)
                        ax.set_xlabel("Padded Image")
                        ax.imshow(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))

                        filter_f = hw2.psf2otf(
                            filter, padded_image.shape[:2]
                        )  # visualize
                        ax = fig.add_subplot(3, 7, 10)
                        ax.set_xticks([])
                        ax.get_yaxis().set_visible(False)
                        ax.set_xlabel("Filter(freq)")
                        ax.imshow(
                            np.log(np.abs(np.fft.fftshift(filter_f))), cmap="gray"
                        )

                        for i, channel in enumerate(cv2.split(padded_image)):
                            channel_f = np.fft.fft2(channel)
                            ax = fig.add_subplot(3, 7, 7 * i + 4)
                            ax.set_xticks([])
                            ax.get_yaxis().set_visible(False)
                            ax.set_xlabel(f"{color_lookup_table[i]}(freq)")
                            ax.imshow(
                                np.log(np.abs(np.fft.fftshift(channel_f))),
                                cmap="gray",
                            )

                            filtered_image_f = filter_f * channel_f  # visualize
                            ax = fig.add_subplot(3, 7, 7 * i + 5)
                            ax.set_xticks([])
                            ax.get_yaxis().set_visible(False)
                            ax.set_xlabel("Low freq")
                            ax.imshow(
                                np.log(np.abs(np.fft.fftshift(filtered_image_f))),
                                cmap="gray",
                            )

                            high_frequency_image_f = (
                                channel_f - filtered_image_f
                            )  # visualize
                            ax = fig.add_subplot(3, 7, 7 * i + 6)
                            ax.set_xticks([])
                            ax.get_yaxis().set_visible(False)
                            ax.set_xlabel("High freq")
                            ax.imshow(
                                np.log(np.abs(np.fft.fftshift(high_frequency_image_f))),
                                cmap="gray",
                            )

                            result_f = channel_f + alpha * high_frequency_image_f
                            ax = fig.add_subplot(3, 7, 7 * i + 7)
                            ax.set_xticks([])
                            ax.get_yaxis().set_visible(False)
                            ax.set_xlabel(f"Result freq")
                            ax.imshow(
                                np.log1p(np.abs(np.fft.fftshift(result_f))),
                                cmap="gray",
                            )

                    fig.tight_layout()

                    process_analysis_file_path = os.path.join(
                        config["output_dir"],
                        domain,
                        f"analysis_{result_image_filename}",
                    )
                    fig.savefig(process_analysis_file_path, format="png")
                    print(
                        f'[ ! ]Saved analysis of unsharpening in {domain} domain at "{process_analysis_file_path}"'
                    )
                    print()


if __name__ == "__main__":
    print("[###]PROBLEM1 IDEAL LOWPASS FILTERING")
    print()
    problem1()

    print("[###]PROBLEM2 GAUSSIAN LOWPASS FILTERING")
    print()
    problem2()

    print("[###]PROBLEM3 CONVOLUTION THEOREM")
    print()
    problem3()
