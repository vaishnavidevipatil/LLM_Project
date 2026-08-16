import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

#######################
brushThickness = 25
eraserThickness = 100
########################


folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
    
print(len(overlayList))
header = overlayList[0]
# drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

# detector = htm.handDetector(detectionCon=0.65,maxHands=1)
# xp, yp = 0, 0
# imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    # 1. Import image
    ret, frame = cap.read()

    img = cv2.imshow(frame, 'Image')
        
    cap.waitKey(1)
    cv2.destroyAllWindows()
    # 2. Find Hand Landmarks
    # img = detector.findHands(img)
    # lmList = detector.findPosition(img, draw=False)

    # if len(lmList) != 0:

    #     # print(lmList)

    #     # tip of index and middle fingers
    #     x1, y1 = lmList[8][1:]
    #     x2, y2 = lmList[12][1:]

    #     # 3. Check which fingers are up
    #     fingers = detector.fingersUp()
    #     # print(fingers)

    #     # 4. If Selection Mode - Two finger are up
    #     if fingers[1] and fingers[2]:
    #         # xp, yp = 0, 0
    #         print("Selection Mode")
    #         # # Checking for the click
    #         if y1 < 125:
    #             if 250 < x1 < 450:
    #                 header = overlayList[0]
    #                 drawColor = (255, 0, 255)
    #             elif 550 < x1 < 750:
    #                 header = overlayList[1]
    #                 drawColor = (255, 0, 0)
    #             elif 800 < x1 < 950:
    #                 header = overlayList[2]
    #                 drawColor = (0, 255, 0)
    #             elif 1050 < x1 < 1200:
    #                 header = overlayList[3]
    #                 drawColor = (0, 0, 0)
    #         cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

    #     # 5. If Drawing Mode - Index finger is up
    #     if fingers[1] and fingers[2] == False:
    #         cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
    #         print("Drawing Mode")
    #         if xp == 0 and yp == 0:
    #             xp, yp = x1, y1

    #         cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

    #         # if drawColor == (0, 0, 0):
    #         #     cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
    #         #     cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
    #         #
    #         # else:
    #         #     cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
    #         #     cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

    #         xp, yp = x1, y1


    #     # # Clear Canvas when all fingers are up
    #     # if all (x >= 1 for x in fingers):
    #     #     imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    # imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    # _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    # imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    # img = cv2.bitwise_and(img,imgInv)
    # img = cv2.bitwise_or(img,imgCanvas)


    # # Setting the header image
    # img[0:125, 0:1280] = header
    # # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    # cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("Inv", imgInv)
    # cv2.waitKey(1)

###############

# import cv2
# import numpy as np
# import os
# import time

# # Suppress TensorFlow Lite & MediaPipe warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import HandTrackingModule as htm  # Ensure this is in the same folder

# # ====== Settings ======
# brushThickness = 15
# eraserThickness = 50
# folderPath = "Header"
# drawColor = (255, 0, 255)
# xp, yp = 0, 0

# # ====== Load Header Images ======
# try:
#     myList = sorted(os.listdir(folderPath))
#     overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in myList if imPath.endswith(('.png', '.jpg'))]
#     header = overlayList[0] if overlayList else np.zeros((125, 1280, 3), np.uint8)
# except Exception as e:
#     print("⚠️ Error loading header images:", e)
#     header = np.zeros((125, 1280, 3), np.uint8)

# # ====== Initialize Webcam ======
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)  # Width
# cap.set(4, 720)   # Height

# success, img = cap.read()
# if success:
#     print("✅ Webcam working.")
# else:
#     print("❌ Webcam failed to open.")
#     exit()

# # ====== Hand Detector ======
# detector = htm.handDetector(detectionCon=0.8, maxHands=1)

# # ====== Canvas ======
# imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# # ====== Main Loop ======
# while True:
#     success, img = cap.read()
#     if not success:
#         print("❌ Failed to grab frame from webcam.")
#         break

#     img = cv2.flip(img, 1)
#     img = detector.findHands(img)
#     lmList = detector.findPosition(img, draw=False)

#     if len(lmList) != 0:
#         x1, y1 = lmList[8][1:]
#         x2, y2 = lmList[12][1:]
#         fingers = detector.fingersUp()

#         # === Selection Mode ===
#         if fingers[1] and fingers[2]:
#             xp, yp = 0, 0
#             if y1 < 125:
#                 if 250 < x1 < 450:
#                     header = overlayList[0]
#                     drawColor = (255, 0, 255)
#                 elif 550 < x1 < 750:
#                     header = overlayList[1]
#                     drawColor = (255, 0, 0)
#                 elif 800 < x1 < 950:
#                     header = overlayList[2]
#                     drawColor = (0, 255, 0)
#                 elif 1050 < x1 < 1200:
#                     header = overlayList[3]
#                     drawColor = (0, 0, 0)
#             cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

#         # === Drawing Mode ===
#         elif fingers[1] and not fingers[2]:
#             cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
#             if xp == 0 and yp == 0:
#                 xp, yp = x1, y1
#             thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
#             cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)
#             cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
#             xp, yp = x1, y1

#         # === Save Drawing if All Fingers Up ===
#         if fingers == [1, 1, 1, 1, 1]:
#             os.makedirs("drawings", exist_ok=True)
#             filename = f"drawings/drawing_{int(time.time())}.png"
#             cv2.imwrite(filename, imgCanvas)
#             print(f"✅ Saved: {filename}")
#             time.sleep(1)

#     # === Combine Canvas and Webcam Image ===
#     imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
#     _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
#     imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
#     img = cv2.bitwise_and(img, imgInv)
#     img = cv2.bitwise_or(img, imgCanvas)

#     # === Overlay Header ===
#     img[0:125, 0:1280] = header

#     # === Show Output Windows ===
#     cv2.imshow("Virtual Painter", img)
#     cv2.imshow("Canvas", imgCanvas)

#     if cv2.waitKey(1) & 0xFF == 27:  # ESC key
#         print("🔴 Exiting Virtual Painter...")
#         break

# # ====== Cleanup ======
# cap.release()
# cv2.destroyAllWindows()
