#!/usr/bin/env python
import cv2
import numpy as np
import math
import pprint
import random

# Increase value to decrease precision and reduce execution time (min : 1) WARN : a scan_scale_factor too high could affect the algorithm's ability to work properly
scan_scale_factor = 1
# This variables define the minimum value required to mark position as candidate with HISTCMP_CORREL method, TIPS : set -1 to produce glitchy picture (recommended value : 0.9)
correlation_minimum = 0.9

img_rgb = cv2.imread("Data-TP/mon.jpg")
template_rgb = cv2.imread("Data-TP/statue.jpg")
img_size = img_rgb.shape[:2]
tmp_size = template_rgb.shape[:2]

template_hist = cv2.calcHist(
    template_rgb, [2], None, [256], [0, 256]
)  # get template histogram
hist_cmp = []

for y in range(0, img_size[0] - tmp_size[0], scan_scale_factor):
    y_upper, y_lower = (y, y + tmp_size[0])
    for x in range(0, img_size[1] - tmp_size[1], scan_scale_factor):
        x_upper, x_lower = (x, x + tmp_size[1])
        print("scanning from {}x{} to {}x{}".format(x_upper, y_upper, x_lower, y_lower))
        a = img_rgb[y_upper:y_lower, x_upper:x_lower]
        img_hist = cv2.calcHist(a, [2], None, [256], [0, 256])
        compared_hist = cv2.compareHist(template_hist, img_hist, cv2.HISTCMP_CORREL)
        hist_cmp.append((compared_hist, (x_upper, y_upper), (x_lower, y_lower)))

pprint.pprint(hist_cmp)
candidates = [
    x for x in hist_cmp if x[0] >= correlation_minimum
]  # max(hist_cmp, key=lambda x: x[0])
print(candidates)

for scope in candidates:
    cv2.rectangle(
        img_rgb,
        scope[1],
        scope[2],
        (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)),
        2,
    )

cv2.imshow("result", img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
