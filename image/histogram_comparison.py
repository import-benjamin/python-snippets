#!/usr/bin/env python
import cv2
import numpy as np
import pprint


img_set = ("beach", "dog", "polar", "bear", "lake", "moose")
img_rgb = {}
img_hist = {}
result = {}

for img_name in img_set:
    img_rgb[img_name] = cv2.imread("Data-TP/{}.jpg".format(img_name), 0)
    print(
        "loaded image : {}, {}x{}px".format(
            img_name, img_rgb[img_name].shape[0], img_rgb[img_name].shape[1]
        )
    )
template_rgb = cv2.imread("Data-TP/waves.jpg", 0)

template_hist = cv2.calcHist(
    template_rgb, [0], None, [256], [0, 256]
)  # get template histogram

for img_name in img_set:
    img_hist["{}".format(img_name)] = cv2.calcHist(
        img_rgb[img_name], [0], None, [256], [0, 256]
    )
    result[img_name] = cv2.compareHist(
        template_hist, img_hist[img_name], cv2.HISTCMP_INTERSECT
    )
    # result[img_name] = dict()
    # result[img_name]["correlation"] = cv2.compareHist(template_hist, img_hist[img_name], cv2.HISTCMP_CORREL)
    # result[img_name]["intersect"] = cv2.compareHist(template_hist, img_hist[img_name], cv2.HISTCMP_INTERSECT)
    # result[img_name]["hellinger"] = cv2.compareHist(template_hist, img_hist[img_name], cv2.HISTCMP_BHATTACHARYYA)
    # result[img_name]["chi-squared"] = cv2.compareHist(template_hist, img_hist[img_name], cv2.HISTCMP_CHISQR)

print(
    "== histogram comparison ==\n{}\n==========================".format(
        pprint.pformat(result)
    )
)

img = list(result.keys())[list(result.values()).index(max(result.values()))]

print(
    "Closest image: (using intersect) -> {} with {} similarity".format(
        img, max(result.values())
    )
)
