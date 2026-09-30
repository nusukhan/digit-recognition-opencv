# Makes a test image with digits (white paper, black digits)
import cv2
import numpy as np

img = np.full((300, 700, 3), 255, dtype=np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
for i, digit in enumerate(["3", "7", "5", "2"]):
    x = 60 + i * 160
    cv2.putText(img, digit, (x, 210), font, 6, (0, 0, 0), 18)

cv2.imwrite("test_digits.png", img)
print("Saved: test_digits.png")