import time
import autopy
from handtracker import HandTracker
import cv2 as cv
import numpy as np

camera = cv.VideoCapture(0)
camera.set(cv.CAP_PROP_FPS, 90)
camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))

tracker = HandTracker()

pTime = 0

smoothing = 1
ploc_x, ploc_y = 0, 0
cloc_x, cloc_y = 0, 0

while True:
    success, frame = camera.read()
    frame = cv.flip(frame, 1)
    frame, hand_landmarks = tracker.findHands(frame, draw=True)
    screenW, screenH = autopy.screen.size()

    h, w, c = frame.shape
    x1, y1 = int(0.1 * w), int(0.1 * h)
    x2, y2 = int(0.7 * w), int(0.6 * h)
    xb, yb = 0, 0
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if hand_landmarks:
        index_finger = hand_landmarks[8]

        if index_finger[2] <= hand_landmarks[7][2]:
            cv.circle(frame, index_finger[1:], 30, (0, 0, 0), cv.FILLED)
            # convert coordinates
            index_x, index_y = index_finger[1:]
            xb, yb = index_x, index_y

            mousex = np.interp(index_x, (x1, x2), (0, screenW + 100))
            mousey = np.interp(index_y, (y1, y2), (0, screenH + 100))
            cloc_x = ploc_x + (mousex - ploc_x) / smoothing
            cloc_y = ploc_y + (mousey - ploc_y) / smoothing

            try:
                autopy.mouse.move(cloc_x, cloc_y)
                ploc_x, ploc_y = cloc_x, cloc_y
            except ValueError:
                print("Points out of bound")

        finger_distance = tracker.find_distance_between_fingers(4, 6, draw_center=False)
        if finger_distance <= 35:
            autopy.mouse.click()

    cTime = time.time()
    fps = round(1 / (cTime - pTime))
    pTime = cTime
    cv.putText(frame, f"{fps} Fps", (10, 30), cv.FONT_ITALIC, 1, (255, 0, 0), 3)

    if cv.waitKey(20) & 0xff == ord("d"):
        break

    frame = cv.resize(frame, (500, 350))
    Bframe = np.zeros(frame.shape, dtype=np.uint8)
    cv.circle(Bframe, (xb, yb), 30, (255, 255, 255), cv.FILLED)

    cv.imshow("Virtual Mouse", frame)
    cv.imshow("Blank ", Bframe)

camera.release()
cv.destroyAllWindows()
