import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

import config
import util

# SIFT(Scale-Invariant Feature Transform)
SIFT = cv2.SIFT_create()

# BF(Brute Force) Matcher
BF = cv2.BFMatcher()


def RANSAC(matches, kp1, kp2, image1, image2) -> None:
    RANSAC_CONFIG = config._RANSAC_CONFIG()

    # Convert matches to kp pair
    matches_kp = [[kp1[match[0].queryIdx], kp2[match[0].trainIdx]] for match in matches]

    target_inlier_count = 0  # maximum inlier count
    target_inlier_pair = []

    for _ in range(RANSAC_CONFIG.N_ITERATION):
        # select random 4 pair of matching point candidates
        random_sample_idx = [random.randint(0, len(matches_kp) - 1) for _ in range(4)]

        src_p = np.asarray(
            [matches_kp[idx][0].pt for idx in random_sample_idx], np.float32
        )
        dst_p = np.asarray(
            [matches_kp[idx][1].pt for idx in random_sample_idx], np.float32
        )

        H = cv2.getPerspectiveTransform(src_p, dst_p)

        inlier = util.find_inlier(H, matches_kp, RANSAC_CONFIG.INLIER_THRESHOLD)
        inlier_count = len(inlier)
        if target_inlier_count < inlier_count:
            target_inlier_count = inlier_count
            target_inlier_pair = inlier

    src = np.asarray([[*p, 1] for p, _ in target_inlier_pair])
    dst = np.asarray([[*p, 1] for _, p in target_inlier_pair])

    # find homography
    # use least square method by specifing method = 0
    # (src: https://docs.opencv.org/4.1.0/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780)
    # they use LMSolver to find solution for least square problem
    hom, _ = cv2.findHomography(src, dst, 0)

    return hom


def stitch_images(
    ref: "np.array[np.uint8 | np.float32]",
    src: "np.array[np.uint8 | np.float32]",
    H: "np.array[np.uint8]",
) -> "np.array[np.float32]":
    ref_height, ref_width, _ = ref.shape
    src_height, src_width, _ = src.shape

    H_Inv = np.linalg.inv(H)

    x1, y1 = util.transform(0, 0, H_Inv)
    x2, y2 = util.transform(0, src_height - 1, H_Inv)
    x3, y3 = util.transform(src_width - 1, 0, H_Inv)
    x4, y4 = util.transform(src_width - 1, src_height - 1, H_Inv)

    min_x = np.round(min(0, x1, x2, x3, x4)).astype(int)
    min_y = np.round(min(0, y1, y2, y3, y4)).astype(int)
    max_x = np.round(max(ref_width, x1, x2, x3, x4)).astype(int)
    max_y = np.round(max(ref_height, y1, y2, y3, y4)).astype(int)

    ofs_x = np.abs(min_x)
    ofs_y = np.abs(min_y)

    height = ofs_y + ref_height + np.abs(ref_height - max_y)
    width = ofs_x + ref_width + np.abs(ref_width - max_x)

    image = np.zeros((height, width, 3))

    # Paste reference image
    image[ofs_y : ofs_y + ref_height, ofs_x : ofs_x + ref_width, :] = ref

    # Generate distance transformation for alpha blending
    # Alpha for blending will be determined by this
    mask = np.zeros((src_width + 2, src_height + 2), np.uint8)
    mask[1 : 1 + src_width, 1 : 1 + src_height] = np.ones((src_width, src_height)) * 255
    _, t = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.distanceTransform(t, cv2.DIST_L2, 5) / 255

    for y in range(min_y, height):
        for x in range(min_x, width):
            if y + ofs_y < height and x + ofs_x < width:
                # Warp the position of pixel
                xp, yp = util.transform(x, y, H)

                if 0 < xp < src_width and 0 < yp < src_height:
                    # Retrieve full pixel
                    pixel = cv2.getRectSubPix(src, (1, 1), (xp, yp))

                    # For non-empty pixels, conduct alpha blending
                    if not np.all(image[y + ofs_y, x + ofs_x] == 0):
                        r = mask[int(xp), int(yp)]
                        image[y + ofs_y, x + ofs_x] = pixel * r + image[
                            y + ofs_y, x + ofs_x
                        ] * (1 - r)
                    # For empty pixels, copy warped pixel
                    else:
                        image[y + ofs_y, x + ofs_x] = pixel

    return image.astype(np.uint8)


def panorama(
    image_list: list["np.ndarray[np.uint8 | np.float32]"],
) -> "np.ndarray[np.uint8 | np.float32]":
    # 1. Given N input images, set one image as a reference
    reference_image = image_list[0]

    for source_image in image_list[1:]:
        # 2. Detect feature points from images and correspondeces between pairs of images
        kp_ref, des_ref = SIFT.detectAndCompute(reference_image, None)
        kp_src, des_src = SIFT.detectAndCompute(source_image, None)

        matches = BF.knnMatch(des_ref, des_src, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        # 3. Estimate the homographics between images using RANSAC
        H = RANSAC(good_matches, kp_ref, kp_src, reference_image, source_image)

        # 4. Warp the images to the reference image
        # 5. Compose them
        reference_image = stitch_images(reference_image, source_image, H)

    return reference_image
