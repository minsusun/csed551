import cv2
import numpy as np


def transform(x: float, y: float, H: "np.ndarray[np.float32]") -> tuple[float, float]:
    """Transform given (x,y) with given Homography H

    Args:
        x (float): x axis value of point
        y (float): y axis value of point
        H (np.ndarray[np.float32]): homography to use

    Returns:
        tuple[float, float]: transformed point (x, y)
    """
    h = np.array([x, y, 1])
    t = H.dot(h)
    w = t[2]
    return t[0] / w, t[1] / w


def match_distance(
    match: list["cv2.KeyPoint", "cv2.KeyPoint"], H: "np.ndarray[np.uint8]"
) -> float:
    """_summary_

    Args:
        match (list[cv2.Keypoint;, cv2.KeyPoint]): pair of keypoints to calculate distance of
        H (np.ndarray[np.uint8]): homography to use when matching each points

    Returns:
        float: distance between two key points after applying the homography on the first key point
    """
    x, y = transform(*match[0].pt, H)
    return np.sqrt((x - match[1].pt[0]) ** 2 + (y - match[1].pt[1]) ** 2)


def find_inlier(
    H: "np.ndarray[np.float32]",
    matches: list[list["cv2.KeyPoint", "cv2.KeyPoint"]],
    threshold: float,
) -> list[int, tuple[float, float], tuple[float, float]]:
    """_summary_

    Args:
        H (np.ndarray[np.float32]): _description_
        matches (list[list[&quot;cv2.KeyPoint&quot;, &quot;cv2.KeyPoint&quot;]]): _description_
        threshold (float): _description_

    Returns:
        list[int, tuple[float, float], tuple[float, float]]: list of (idx of match, point, point) inling with the threshold
    """
    inlier = [
        [idx, match[0].pt, match[1].pt]
        for idx, match in enumerate(matches)
        if match_distance(match, H) < threshold
    ]
    return inlier
