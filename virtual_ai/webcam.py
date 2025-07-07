import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm


# #######################
# brushThickness = 25
# eraserThickness = 100
# ########################


folderPath = "header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
    
print(len(overlayList))
header = overlayList[0]
print(header.shape)
drawColor = (255, 0, 255) # Purple color for drawing

# Select the 4th image (index 3) as header (i.e., 4.jpg)
header = overlayList[3]  # index starts from 0, so 3 = "4.jpg"

# Resize header to match webcam width
header = cv2.resize(header, (1280, 125))  # 125 px height toolbar
drawColor = (255, 0, 255) # Purple color for drawing

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
        
    # Resize webcam feed to match width
    frame = cv2.resize(frame, (1280, 720))

    # Overlay the header on top of the webcam feed
    frame[0:125, 0:1280] = header  # Replace top 125px with header

    cv2.imshow('Webcam Test', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# cap.release()
# cv2.destroyAllWindows()

# Test reading an image using OpenCV
path = "C:/Users/vaish/code/LLM_Project/virtual_ai/4.jpg"   
img = cv2.imread(path)
print(img.shape)
cv2.imshow('Test Image', img)