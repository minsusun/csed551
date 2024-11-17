import cv2

import hw3

image_magic_string = "../images/input/Rainier{}.png"
image_result_path = "../images/output/stitched.png"
image_n = 5

additional_image_magic_string = "../images/input/addition.png"
abnormal_result_path = image_result_path = "../images/output/abnormal.png"


def stitch_the_images():
    image_list = []
    for idx in range(1, image_n):
        image_list.append(cv2.imread(image_magic_string.format(idx)))
    result = hw3.panorama(image_list)
    cv2.imwrite(image_result_path, result)


def stitch_the_abnormal():
    image_list = []
    for idx in range(1, image_n):
        image_list.append(cv2.imread(image_magic_string.format(idx)))
    image_list.append(cv2.imread(additional_image_magic_string))
    result = hw3.panorama(image_list)
    cv2.imwrite(abnormal_result_path, result)

def stitch_the_cal():
    image_list = []
    for idx in range(1, 3):
        image_list.append(cv2.imread(f"../images/input/cal{idx}.JPG"))
    result = hw3.panorama(image_list)
    cv2.imwrite("../images/output/cal.png", result)

if __name__ == "__main__":
    # stitch_the_images()
    # stitch_the_abnormal()
    stitch_the_cal()
