import cv2

bgr = cv2.imread("Data-TP/bgr.png")

bw_only_blue = bgr.copy()[:, :, 0]
bw_only_green = bgr.copy()[:, :, 1]
bw_only_red = bgr.copy()[:, :, 2]

bgr_without_blue = bgr.copy()
bgr_without_blue[:, :, 0] = 0

bgr_without_green = bgr.copy()
bgr_without_green[:, :, 1] = 0

bgr_without_red = bgr.copy()
bgr_without_red[:, :, 2] = 0

bgr_only_blue = bgr_without_green.copy()
bgr_only_blue[:, :, 2] = 0

bgr_only_green = bgr_without_red.copy()
bgr_only_green[:, :, 0] = 0

bgr_only_red = bgr_without_green.copy()
bgr_only_red[:, :, 0] = 0

cv2.imshow("bgr", bgr)
cv2.imshow("bgr_blue", bw_only_blue)
cv2.imshow("bgr_green", bw_only_green)
cv2.imshow("bgr_red", bw_only_red)

cv2.imshow("bgr_without_blue", bgr_without_blue)
cv2.imshow("bgr_without_gree", bgr_without_green)
cv2.imshow("bgr_without_red", bgr_without_red)

cv2.imshow("bgr_only_red", bgr_only_red)
cv2.imshow("bgr_only_green", bgr_only_green)
cv2.imshow("bgr_only_blue", bgr_only_blue)


cv2.waitKey(0)
cv2.destroyAllWindows()
