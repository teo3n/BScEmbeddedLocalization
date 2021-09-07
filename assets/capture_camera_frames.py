import cv2


cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


f_id = 0

while True:
	img = cap.read()[1]


	cv2.imshow("img", img)
	key = cv2.waitKey(1)

	if key == ord('r'):
		cv2.imwrite("calib/checker_" + str(f_id) + ".png", img)
		f_id += 1
	elif key == ord('q'):
		break
