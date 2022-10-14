import cv2
import numpy as np
import pytesseract

# Read input image, convert to grayscale
img = cv2.imread('NiVUK.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Remove shadows, cf. https://stackoverflow.com/a/44752405/11089932
dilated_img = cv2.dilate(gray, np.ones((7, 7), np.uint8))
bg_img = cv2.medianBlur(dilated_img, 21)
diff_img = 255 - cv2.absdiff(gray, bg_img)
norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# Threshold using Otsu's
work_img = cv2.threshold(norm_img, 0, 255, cv2.THRESH_OTSU)[1]

# Tesseract
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(work_img, config=custom_config)
print(text)
