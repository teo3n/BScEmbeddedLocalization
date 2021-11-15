import numpy as np
import cv2

cap = cv2.VideoCapture(4)

ref_img = cap.read()[1]
for i in range(50):
    ref_img = cap.read()[1]

while True:
    img = cap.read()[1]

    l_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(16, 15)
    disparity = stereo.compute(l_gray, r_gray)

    disparity = cv2.convertScaleAbs(disparity)

    cv2.imshow("img", disparity)
    cv2.imshow("stereo", np.hstack((ref_img, img)))
    cv2.waitKey(1)

