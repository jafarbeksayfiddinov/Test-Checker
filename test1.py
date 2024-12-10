
from cryptography.fernet import Fernet
import qrcode
import cv2

fixed_key = b"3QD_AaN0Z4m_AU1qPUgSmv9fUwr2V-g-8hLx0NV3o1w="  # Replace with your fixed key
cipher = Fernet(fixed_key)

#
# def generate_qr_code(message, qr_path="secret_message.png"):
#     encrypted_message = cipher.encrypt(message.encode())
#     print(f"Encrypted Message: {encrypted_message.decode()}")
#
#     qr = qrcode.QRCode(
#         version=1,
#         error_correction=qrcode.constants.ERROR_CORRECT_L,
#         box_size=10,
#         border=4,
#     )
#     qr.add_data(encrypted_message.decode())
#     qr.make(fit=True)
#
#     img = qr.make_image(fill_color="black", back_color="white")
#     img.save(qr_path)
#     print(f"QR code saved as {qr_path}")
#

def decode_qr_code(qr_path="revised_photo.png"):
    image = cv2.imread(qr_path)
    detector = cv2.QRCodeDetector()

    decrypted_data, points, _ = detector.detectAndDecode(image)
    print(points[0])
    for i in points[0]:
        print((int)(i[0]),(int)(i[1]))
    # print(points)
    if not decrypted_data:
        print("No QR code found in the image.")
        return

    print(f"Encrypted Message from QR Code: {decrypted_data}")

    decrypted_message = cipher.decrypt(decrypted_data.encode())
    print(f"Decrypted Message: {decrypted_message.decode()}")


if __name__ == "__main__":
    # Encrypt and generate QR code
    # generate_qr_code("1.A 2.B 3.C")

    # Decode the QR code and decrypt the message
    decode_qr_code()
