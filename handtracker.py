import math

import cv2 as cv
import mediapipe as mp


class HandTracker:
    def __init__(self, max_hands=1, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.max_hands = max_hands
        self.mode = static_image_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(max_num_hands=self.max_hands,
                                        static_image_mode=self.mode,
                                        min_detection_confidence=self.min_detection_confidence,
                                        min_tracking_confidence=self.min_tracking_confidence, )

    def findHands(self, image, draw=True):
        self.frame = image
        self.frameRGB = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(self.frameRGB)
        self.draw = draw
        self.landmark_list = []

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for point, lm in enumerate(handLms.landmark):
                    h, w, c = self.frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.landmark_list.append([point, cx, cy])

                if self.draw:
                    self.mpDraw.draw_landmarks(
                        self.frame, handLms, self.mpHands.HAND_CONNECTIONS,
                        connection_drawing_spec=self.mpDraw.DrawingSpec(color=self.mpDraw.RED_COLOR, thickness=4),
                        landmark_drawing_spec=self.mpDraw.DrawingSpec(color=self.mpDraw.RED_COLOR, thickness=2)
                    )
        return self.frame, self.landmark_list

    def find_distance_between_fingers(self, finger1, finger2, draw_center=True):
        x1, y1 = self.landmark_list[finger1][1:]
        x2, y2 = self.landmark_list[finger2][1:]

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        finger_dist = math.hypot(x2 - x1, y2 - y1)

        if draw_center:
            cv.line(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv.circle(self.frame, (cx, cy), 10, (255, 0, 0), cv.FILLED)

        return finger_dist


def main():
    cap = cv.VideoCapture(0)
    detector = HandTracker()

    while True:
        _, frame = cap.read()
        frame = cv.flip(frame, 1)
        frame, lmList = detector.findHands(frame)
        print(lmList)

        cv.imshow("Image", frame)

        if cv.waitKey(1) & 0xff == ord("d"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
