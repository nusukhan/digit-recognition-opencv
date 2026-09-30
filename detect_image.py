# ============================================================
#  detect_image.py  —  Digit detection on an image (OpenCV)
#
#  Full OpenCV pipeline on a picture instead of a live camera:
#     load image -> grayscale -> blur -> threshold -> find
#     contours -> crop each digit -> predict with the SVM model
#     -> draw a green box + the predicted number
#
#  Output: result.png  (the image with boxes and predictions)
#
#  Run:  python detect_image.py test_digits.png
# ============================================================

import sys
import cv2
import joblib
import numpy as np

image_path = sys.argv[1] if len(sys.argv) > 1 else "test_digits.png"

bundle = joblib.load("model.pkl")
model, labels = bundle["model"], bundle["labels"]

img = cv2.imread(image_path)
if img is None:
    sys.exit(f"Could not read image: {image_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         # 1. to grayscale
blur = cv2.GaussianBlur(gray, (5, 5), 0)             # 2. smooth out noise
_, thresh = cv2.threshold(blur, 0, 255,              # 3. threshold (invert)
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,   # 4. find shapes
                               cv2.CHAIN_APPROX_SIMPLE)

found = 0
for cnt in contours:
    if cv2.contourArea(cnt) < 300:      # ignore tiny specks
        continue

    x, y, w, h = cv2.boundingRect(cnt)  # box around this shape
    if not (0.2 < w / h < 1.5):         # skip non-digit shapes
        continue

    pad = 10                             # 5. crop + resize to 8x8
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, thresh.shape[1]), min(y + h + pad, thresh.shape[0])
    roi = thresh[y0:y1, x0:x1]
    small = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_AREA)
    feat = (small.astype(np.float32) / 255.0 * 16.0).reshape(1, -1)

    pred = model.predict(feat)[0]        # 6. predict and draw
    label = labels[pred]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 3)
    cv2.putText(img, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 3)
    found += 1

cv2.imwrite("result.png", img)
print(f"Detected {found} digit(s). Saved: result.png")