import numpy as np
import easyocr
import cv2
import pytesseract

# import pytesseract
# from PIL import Image

image = cv2.imread('data sample.png')
# pil_img=Image.open('table sample.png')


reader = easyocr.Reader(['en'], gpu=False)


def line_sorting(line):
    print(line)
    x1, y1, x2, y2 = line[0]
    return min(x1 + y1, x2 + y2)


vertical_lines = []
horizontal_lines = []

min_raw_distance = [400000]
max_raw_distance = [0]

min_column_distance = [40000]
max_column_distance = [0]

mid_row_y = 0;


def vertical_sorting(line):
    x1, y1, x2, y2 = line
    return min(x1, x2)


def horizontal_sorting(line):
    x1, y1, x2, y2 = line
    return min(y1, y2)


def detect_all_lines(points):
    x1, y1, x2, y2 = points[0]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if y1 >= 600 and y2 >= 600:
        if x1 == x2:
            # x=0
            vertical_lines.append([int(x1), int(y1), int(x2), int(y2)])

            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lines_list.append([(x1, y1), (x2, y2)])

        elif y1 == y2:
            if x2 - x1 >= 400:
                horizontal_lines.append([int(x1), int(y1), int(x2), int(y2)])
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                lines_list.append([(x1, y1), (x2, y2)])


def calculating_column_distance(vertical_lines):
    for coordinate in vertical_lines:
        x1, y1, x2, y2 = coordinate
        d = y2 - y1
        min_column_distance[0] = min(min_column_distance[0], d)
        max_column_distance[0] = max(max_column_distance[0], d)



def calculating_row_distance(horizontal_lines):
    for coordinate in horizontal_lines:
        x1, y1, x2, y2 = coordinate
        d = x2 - x1
        min_raw_distance[0] = min(min_raw_distance[0], d)
        max_raw_distance[0] = max(max_raw_distance[0], d)


def recognizing_text(path):
    # Load the image
    image = cv2.imread(path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for better contrast
    binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

    # Optionally, invert the image if text is black on a white background
    inverted_image = cv2.bitwise_not(binary_image)

    # Perform OCR with Tesseract
    config = "--oem 3 --psm 10 -c tessedit_char_whitelist=ABCD0123456789"  # Alphanumeric whitelist
    text = pytesseract.image_to_string(inverted_image, config=config)
    return text.strip()


# Read image

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines_list = []
lines = cv2.HoughLinesP(
    edges,  # Input edge image
    1,  # Distance resolution in pixels
    np.pi / 180,  # Angle resolution in radians
    threshold=100,  # Min number of votes for valid line
    minLineLength=5,  # Min allowed length of line
    maxLineGap=10  # Max allowed gap between line for joining them
)

for points in lines:
    detect_all_lines(points)

cv2.imwrite('table_detected.png', image)
vertical_lines = sorted(vertical_lines, key=vertical_sorting)
horizontal_lines = sorted(horizontal_lines, key=horizontal_sorting)


def mid_row_calculation_y():
    sum=0
    for horizontal_line in horizontal_lines:
        sum+=horizontal_line[1]
    return sum//len(horizontal_lines)


def removing_duplicate_verticals(vertical_lines):
    new_list = [vertical_lines[0]]
    for i in range(1,len(vertical_lines)):
        x1_0, y1_0, x2_0, y2_0 = vertical_lines[i]
        x1_1, y1_1, x2_1, y2_1 = vertical_lines[i - 1]
        if not (abs(x1_0 - x1_1) <= 10 ):
            new_list.append(vertical_lines[i])

    return new_list


def removing_duplicate_horizontals(horizontal_lines):
    new_list = [horizontal_lines[0]]
    for i in range(1, len(horizontal_lines)):
        x1_0, y1_0, x2_0, y2_0 = horizontal_lines[i]
        x1_1, y1_1, x2_1, y2_1 = horizontal_lines[i - 1]
        if not (abs(y1_0 - y1_1) <= 10 ):
            new_list.append(horizontal_lines[i])

    return new_list


horizontal_lines=removing_duplicate_horizontals(horizontal_lines)
vertical_lines=removing_duplicate_verticals(vertical_lines)
mid_row_y = mid_row_calculation_y()

calculating_column_distance(vertical_lines)

calculating_row_distance(horizontal_lines)
qn = 1
# print(horizontal_lines)
# print(len(vertical_lines))
for i in range(len(vertical_lines) - 1):
    x1_0, y1_0, x2_0, y2_0 = vertical_lines[i]
    x1_1, y1_1, x2_1, y2_1 = vertical_lines[i + 1]
    # print(mid_row_y, x1_0, x1_1, y1_0, y1_1)
    crop1 = image[y1_0 + 5:mid_row_y - 5, x1_0 + 5:x1_1-5]
    crop2 = image[mid_row_y + 5:y2_1 - 5, x1_0 + 5:x1_1-5]

    result1 = reader.readtext(image=crop1)
    result2 = reader.readtext(image=crop2)
    # pil_crop1 = Image.fromarray(cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB))
    # pil_crop2 = Image.fromarray(cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB))

    # text1 = pytesseract.image_to_string(pil_crop1)
    # text2 = pytesseract.image_to_string(pil_crop2)

    # if len(result2)>=1:
    #     print(f"{result2[0][1]}",end=" , ")
    # print(text1,text2,end=" | ")
    cv2.imwrite(f"crop{qn}|1.jpg", crop1)
    cv2.imwrite(f"crop{qn}|2.jpg", crop2)
    text1 = recognizing_text(f"crop{qn}|1.jpg")
    text2 = recognizing_text(f"crop{qn}|2.jpg")
    qn += 1
    print(text1, text2, end=" , ")

# print(min_x, min_y)
# print(vertical_lines)
# print(len(vertical_lines))
#
# print(horizontal_lines)
# print(len(horizontal_lines))
#
# print(min_column_distance, max_column_distance)
# print(min_raw_distance, max_raw_distance)
