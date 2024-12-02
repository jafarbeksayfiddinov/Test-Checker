import cv2
import pytesseract

# Load the image
image = cv2.imread("crop3|1.jpg")

# Preprocess: Grayscale, Resize, Adaptive Threshold
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
adaptive_thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# OCR Configuration
config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(adaptive_thresh, config=config)

print("Extracted Text:")
print(text)
