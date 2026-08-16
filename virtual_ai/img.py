# # import os
# # import cv2

# # # ✅ Use correct relative path (your image is in "Header" folder)
# # path = os.path.join("Header", "4.jpg")

# # # ✅ Check if file exists before reading
# # print("Path exists:", os.path.exists(path))

# # # ✅ Read the image
# # img = cv2.imread(path)

# # # ✅ Handle image loading properly
# # if img is None:
# #     print("❌ cv2 can't read the image. Check path or file type.")
# # else:
# #     print("✅ Image shape:", img.shape)
# #     cv2.imshow("Test Image", img)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()


# import cv2
# import os

# # Path to header image (e.g., "Header/1.jpg")
# header_path = os.path.join("Header", "1.jpg")
# header = cv2.imread(header_path)

# # Resize header to match webcam width (e.g., 1280px wide, 125px tall)
# header_height = 125
# header_width = 1280
# header = cv2.resize(header, (header_width, header_height))

# # Start webcam
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, header_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Resize webcam frame to match height minus header
#     frame = cv2.resize(frame, (header_width, 720))

#     # Combine header + webcam vertically using numpy
#     full_frame = cv2.vconcat([header, frame])

#     # Show the combined frame
#     cv2.imshow("Virtual Painter", full_frame)

#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# img.py  –  Virtual Painter demo
import cv2
import numpy as np
import os
from HandTrackingModule import handDetector   # ← module from earlier reply

# ------------------------------------------------------------
# 1.  Load toolbar / header images
folderPath = "header"
imgFiles   = sorted(os.listdir(folderPath))   # ensure order 1.jpg … 4.jpg
overlayList = [cv2.imread(os.path.join(folderPath, f)) for f in imgFiles]

header_id      = 0               # start with first header
header_height  = 125             # toolbar height
header_width   = 1280            # webcam width
header         = cv2.resize(overlayList[header_id], (header_width, header_height))

# Map each header strip to a drawing colour
# (B,G,R) – adjust if your images are arranged differently
palette = [(255, 0,255),          # 1.jpg  → purple
           (255, 0,  0),          # 2.jpg  → blue
           (  0,255,  0),         # 3.jpg  → green
           (  0, 0,  0)]          # 4.jpg  → black (eraser)

drawColor      = palette[0]
brushThickness = 15
xp, yp         = 0, 0            # previous coords for drawing
imgCanvas      = np.zeros((720, header_width, 3), np.uint8)

# ------------------------------------------------------------
# 2.  Init webcam & hand detector
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  header_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = handDetector(detectionCon=0.7, maxHands=1)

# ------------------------------------------------------------
# 3.  Main loop
while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)                       # mirror‑view
    img = detector.findHands(img, draw=True)
    lmList, _ = detector.findPosition(img, draw=False)

    if lmList:
        # Index and middle‑finger tips
        x1, y1 = lmList[8][1:]      # index finger tip
        x2, y2 = lmList[12][1:]     # middle finger tip

        fingers = detector.fingersUp()           # [thumb,idx,mid,ring,pinky]

        # ---------------- Selection mode  (index & middle fingers up) -----
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0                       # reset previous points
            # If hand is in toolbar region
            if y1 < header_height:
                # Choose colour block based on x‑position
                if   250 < x1 < 450:
                    header_id, drawColor = 0, palette[0]
                elif 550 < x1 < 750:
                    header_id, drawColor = 1, palette[1]
                elif 800 < x1 < 950:
                    header_id, drawColor = 2, palette[2]
                elif 1050 < x1 < 1200:
                    header_id, drawColor = 3, palette[3]
                header = cv2.resize(overlayList[header_id],
                                     (header_width, header_height))

            # Visual feedback rectangle
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)

        # ---------------- Drawing mode  (only index finger up) -------------
        elif fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1          # first point
            # Draw on both live frame & permanent canvas
            cv2.line(img,       (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    # ------------------------------------------------------------
    # 4.  Combine live feed with the canvas so old strokes persist
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # 5.  Overlay toolbar on top
    img[0:header_height, 0:header_width] = header

    cv2.imshow("Virtual Painter", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
