# ISP PipeLine (RAW image -> JPEG image)
# 1. White Balancing(Automatic White Balance, AWB), Normalized
# 2. CFA Interpolation, Normalized
# 3. Sharpening, np.uint8
# 4. Saturation Enhancement, np.uint8
# 5. Gamma Correction, np.uint8

import sys

import cv2
import exifread
import numpy as np


def awb(image, cfa_type):
    # Auto White Balance
    height, width = image.shape

    R = []
    G = []
    B = []

    # gather pixels on color filter
    for i in range(height):
        for j in range(width):
            filter = cfa_type[i % 2][j % 2]
            if filter == "R":
                R.append(image[i][j])
            elif filter == "G":
                G.append(image[i][j])
            elif filter == "B":
                B.append(image[i][j])

    # mean values of each color pixels
    R_avg = np.mean(R)
    G_avg = np.mean(G)
    B_avg = np.mean(B)

    # coefficient for auto white balancing
    R_coef = G_avg / R_avg
    B_coef = G_avg / B_avg

    result = image.copy()

    for i in range(height):
        for j in range(width):
            filter = cfa_type[i % 2][j % 2]
            if filter == "R":
                result[i][j] = result[i][j] * R_coef
            elif filter == "B":
                result[i][j] = result[i][j] * B_coef

    return np.clip(result, 0, 1)


def cfa_interpolation(image, cfa_type):
    height, width = image.shape
    result = np.zeros((height, width, 3))  # RGB channel

    def get_filter(i, j):
        return cfa_type[i % 2][j % 2]

    def neighbor_mean(i, j, target):
        assert target in ["R", "G", "B"], f"Unkown target: {target}"

        directions = [
            [1, 1], [1, -1], [-1, 1], [-1, -1],
            [1, 0], [0, 1], [-1, 0], [0, -1],
        ]
        neighbor = []

        for direction in directions:
            pos = (i + direction[0], j + direction[1])

            if not (0 <= pos[0] < height and 0 <= pos[1] < width):
                continue

            if target == get_filter(pos[0], pos[1]):
                neighbor.append(image[pos[0]][pos[1]])

        return np.mean(neighbor)

    for i in range(height):
        for j in range(width):
            filter = get_filter(i, j)
            if filter == "R":
                r = image[i][j]
                g = neighbor_mean(i, j, "G")
                b = neighbor_mean(i, j, "B")
            elif filter == "G":
                r = neighbor_mean(i, j, "R")
                g = image[i][j]
                b = neighbor_mean(i, j, "B")
            elif filter == "B":
                r = neighbor_mean(i, j, "R")
                g = neighbor_mean(i, j, "G")
                b = image[i][j]
            result[i][j] = [b, g, r]  # fuck opencv why does it have to be BGR, not RGB

    return np.clip(result, 0, 1)


def sharpening(image, intensity):
    # TODO: use HSI color space sharpening
    blurred_image = cv2.bilateralFilter(image, 9, 75, 75)
    result = image.copy()
    result = image + intensity * (image - blurred_image)
    return np.clip(result, 0, 255).astype(np.uint8)


def saturation(image, intensity):
    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    delta = np.full(image.shape, (0, intensity, 0), dtype=np.uint8)
    result = np.add(image_HSV, delta).astype(np.uint8)
    return cv2.cvtColor(np.clip(result, 0, 255), cv2.COLOR_HSV2BGR)


def gamma_correction(image, gamma):
    result = image.copy() / 255
    result = result ** (1 / gamma) * 255
    return np.clip(result, 0, 255).astype(np.uint8)


def color_temp(image, target_temp):
    kelvin_table = {
        1000: (255, 56, 0),
        1500: (255, 109, 0),
        2000: (255, 137, 18),
        2500: (255, 161, 72),
        3000: (255, 180, 107),
        3500: (255, 196, 137),
        4000: (255, 209, 163),
        4500: (255, 219, 186),
        5000: (255, 228, 206),
        5500: (255, 236, 224),
        6000: (255, 243, 239),
        6500: (255, 249, 253),
        7000: (245, 243, 255),
        7500: (235, 238, 255),
        8000: (227, 233, 255),
        8500: (220, 229, 255),
        9000: (214, 225, 255),
        9500: (208, 222, 255),
        10000: (204, 219, 255)}
    r, g, b = kelvin_table[target_temp]
    matrix = np.array([
        [b / 255, 0, 0],
        [0, g / 255, 0],
        [0, 0, r / 255]
    ])
    result = np.matmul(image, matrix)
    return np.clip(result, 0, 255).astype(np.uint8)

def bright(image, intensity):
    result = image + intensity * image
    return np.clip(result, 0, 255).astype(np.uint8)

def main():
    # use cv2 to open .tiff file
    # use specific flag to reserve all the bits
    tiff_file_name = sys.argv[1]
    image = (
        cv2.imread(tiff_file_name, cv2.IMREAD_UNCHANGED) / np.iinfo(np.uint16).max
    )  # normalize [0, 1]

    # TODO: exif data -> get CFA config; tldr; given images are from samsung S7 Edge
    exif = exifread.process_file(open(tiff_file_name, "rb"))

    # height, width = image.shape
    # typically GRBG, BGGR when rotated
    cfa_type = (
        [["G", "R"], ["B", "G"]]
        if image.shape[0] < image.shape[1]
        else [["B", "G"], ["G", "R"]]
    )

    aux_1 = awb(image, cfa_type)
    aux_2 = (cfa_interpolation(aux_1, cfa_type) * 255).astype(np.uint8)
    aux_3 = sharpening(aux_2, 1)
    aux_4 = saturation(aux_3, 30)
    aux_5 = gamma_correction(aux_4, 2.8)
    aux_6 = color_temp(aux_5, 5500)

    aux_list = [aux_2, aux_3, aux_4, aux_5, aux_6]

    file_name = tiff_file_name.split("/")[-1].split(".")[:-1][0]
    for idx, aux in enumerate(aux_list[:-1]):
        cv2.imwrite(
            f"images/result/{file_name}_aux_{idx}.jpg",
            aux,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )
    cv2.imwrite(f"images/result/{file_name}.jpg", aux_list[-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == "__main__":
    main()
