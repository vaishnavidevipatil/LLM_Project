import cv2
import mediapipe as mp
import numpy as np

# Setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

canvas = np.zeros((720, 1280, 3), np.uint8)
draw_color = (255, 0, 255)
xp, yp = 0, 0

def fingers_up(lmList):
    tips = [8, 12]  # Index and middle finger tips
    fingers = []

    for tip in tips:
        if lmList[tip][2] < lmList[tip - 2][2]:  # If tip is above pip joint
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if lmList:
                x1, y1 = lmList[8][1:]  # Index finger
                x2, y2 = lmList[12][1:]  # Middle finger

                fingers = fingers_up(lmList)

                if fingers[0] and not fingers[1]:  # Only index finger up
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    cv2.line(canvas, (xp, yp), (x1, y1), draw_color, 10)
                    xp, yp = x1, y1
                elif fingers[0] and fingers[1]:  # Both up = mode change
                    xp, yp = 0, 0
                    cv2.rectangle(img, (x1, y1 - 20), (x2, y2 + 20), draw_color, cv2.FILLED)

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Virtual Painter", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
