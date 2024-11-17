import numpy as np


def transform(x, y, H):
    h = np.array([x, y, 1])
    t = H.dot(h)
    w = t[2]
    return t[0] / w, t[1] / w


def match_distance(match, H):
    x, y = transform(*match[0].pt, H)
    return np.sqrt((x - match[1].pt[0]) ** 2 + (y - match[1].pt[1]) ** 2)


def find_inlier(H, matches, threshold):
    inlier = [
        [match[0].pt, match[1].pt]
        for match in matches
        if match_distance(match, H) < threshold
    ]
    return inlier
