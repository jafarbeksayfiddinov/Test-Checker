import cv2
import numpy as np
from sympy import Integer

# Read image
image = cv2.imread('unnecessary/sudoku.png')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply HoughLinesP method to
# to directly obtain line end points
lines_list = []
lines = cv2.HoughLinesP(
    edges,  # Input edge image
    1,  # Distance resolution in pixels
    np.pi / 180,  # Angle resolution in radians
    threshold=100,  # Min number of votes for valid line
    minLineLength=5,  # Min allowed length of line
    maxLineGap=10  # Max allowed gap between line for joining them
)
print(lines)
# Iterate over points
min_x,min_y=40000,40000
for points in lines:
    # Extracted points nested in the list
    x1, y1, x2, y2 = points[0]
    if(x1<=min_x and y1<=min_y):
        min_x=x1
        min_y=y1
    # Draw the lines joing the points
    # On the original image
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Maintain a simples lookup list for points
    lines_list.append([(x1, y1), (x2, y2)])

# Save the result image
cv2.imwrite('unnecessary/detectedLines.png', image)
print(min_x,min_y)