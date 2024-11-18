import cv2

import hw3

image_magic_string = "../images/input/Rainier{}.png"
image_result_path = "../images/output/stitched.png"
image_progress_path = "../images/output/progress_{}.png"
image_n = 5

additional_image_magic_string = "../images/input/addition.png"
abnormal_result_path = image_result_path = "../images/output/abnormal.png"


def stitch_the_images():
    image_list = []
    for idx in range(1, image_n):
        image_list.append(cv2.imread(image_magic_string.format(idx)))
    result, progress_list = hw3.panorama(image_list)
    cv2.imwrite(image_result_path, result)
    for idx, image in enumerate(progress_list):
        cv2.imwrite(image_progress_path.format(idx), image)


def stitch_the_abnormal():
    image_list = []
    for idx in range(1, image_n):
        image_list.append(cv2.imread(image_magic_string.format(idx)))
    image_list.append(cv2.imread(additional_image_magic_string))
    result = hw3.panorama(image_list)
    cv2.imwrite(abnormal_result_path, result)

if __name__ == "__main__":
    stitch_the_images()
    # stitch_the_abnormal()
