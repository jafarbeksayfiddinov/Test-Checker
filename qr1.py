import cv2
from pyzbar.pyzbar import decode

# Step 1: Load the QR code image
image_path = "secret_message.png"  # Path to the QR code image
image = cv2.imread(image_path)

# Step 2: Decode the QR code
decoded_objects = decode(image)

# Step 3: Extract and print the text
if decoded_objects:
    for obj in decoded_objects:
        print("Decoded Text:", obj.data.decode())  # Convert bytes to string
else:
    print("No QR code detected in the image.")
