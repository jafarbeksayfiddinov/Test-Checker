from PIL import Image, ImageEnhance, ImageOps
import pytesseract

# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'  # Update for your setup

# Load the image
image_path = 'data sample.png'  # Replace with your image path
image = Image.open(image_path)

# Step 1: Crop the bottom table area (adjust crop values if needed)
width, height = image.size
table_area = image.crop((0, int(height * 0.8), width, height))  # Bottom 20% of the image
table_area.save('debug_cropped_table.png')  # Save cropped image for debugging

# Step 2: Preprocess the image
gray_image = ImageOps.grayscale(table_area)  # Convert to grayscale
threshold_image = gray_image.point(lambda x: 0 if x < 140 else 255, '1')  # Apply binary threshold
threshold_image.save('debug_processed_table.png')  # Save processed image for debugging

# Step 3: Extract text using OCR
custom_config = r'--psm 6'  # Assume uniform block of text
table_text = pytesseract.image_to_string(threshold_image, config=custom_config)

# Step 4: Debug and process extracted text
print("Raw OCR Output:")
print(table_text)  # Print the raw output to debug
lines = table_text.split('\n')
table_values = [line.strip() for line in lines if line.strip()]  # Remove empty lines

# Step 5: Output the results
print("Extracted Table Values:")
for line in table_values:
    print(line)
